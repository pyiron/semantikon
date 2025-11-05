import ast
import builtins
import copy
import dataclasses
import inspect
import textwrap
from collections import deque
from functools import cached_property, update_wrapper
from typing import Any, Callable, Generic, Iterable, TypeVar, cast, get_args, get_origin

import networkx as nx
from networkx.algorithms.dag import topological_sort

from flowrep import workflow as fwf
from semantikon.converter import (
    get_annotated_type_hints,
    get_return_expressions,
    get_return_labels,
    meta_to_dict,
    parse_input_args,
    parse_output_args,
)
from semantikon.datastructure import (
    MISSING,
    CoreMetadata,
    Edges,
    Function,
    Input,
    Inputs,
    Missing,
    Nodes,
    Output,
    Outputs,
    PortType,
    TypeMetadata,
    Workflow,
)


class SemantikonFunctionWithWorkflow(fwf.FunctionWithWorkflow):
    def run(self, *args, **kwargs) -> dict[str, Any]:
        return super().run(with_function=True, *args, **kwargs)

    def serialize_workflow(self) -> dict:
        wf_dict = self._serialize_workflow(with_function=True, with_outputs=True)
        return to_semantikon_workflow_dict(wf_dict)


def separate_types(
    data: dict[str, Any], class_dict: dict[str, type] | None = None
) -> tuple[dict[str, Any], dict[str, type]]:
    """
    Separate types from the data dictionary and store them in a class dictionary.
    The types inside the data dictionary will be replaced by their name (which
    would for example make it easier to hash it).

    Args:
        data (dict[str, Any]): The data dictionary containing nodes and types.
        class_dict (dict[str, type], optional): A dictionary to store types. It
            is mainly used due to the recursivity of this function. Defaults to
            None.

    Returns:
        tuple: A tuple containing the modified data dictionary and the
            class dictionary.
    """
    data = copy.deepcopy(data)
    if class_dict is None:
        class_dict = {}
    if "nodes" in data:
        for key, node in data["nodes"].items():
            child_node, child_class_dict = separate_types(node, class_dict)
            class_dict.update(child_class_dict)
            data["nodes"][key] = child_node
    for io_ in ["inputs", "outputs"]:
        for key, content in data[io_].items():
            if "dtype" in content and isinstance(content["dtype"], type):
                class_dict[content["dtype"].__name__] = content["dtype"]
                data[io_][key]["dtype"] = content["dtype"].__name__
    return data, class_dict


def _get_node_outputs(func: Callable, counts: int | None = None) -> dict[str, dict]:
    output_hints = parse_output_args(
        func, separate_tuple=(counts is None or counts > 1)
    )
    output_vars = get_return_expressions(func)
    if output_vars is None or len(output_vars) == 0:
        return {}
    if (counts is not None and counts == 1) or isinstance(output_vars, str):
        if isinstance(output_vars, str):
            return {output_vars: cast(dict, output_hints)}
        else:
            return {"output": cast(dict, output_hints)}
    assert isinstance(output_vars, tuple), output_vars
    assert counts is None or len(output_vars) == counts, output_vars
    if output_hints == {}:
        return {key: {} for key in output_vars}
    else:
        assert counts is None or len(output_hints) == counts
        return {key: hint for key, hint in zip(output_vars, output_hints)}


def get_node_dict(
    function: Callable,
    inputs: dict[str, dict] | None = None,
    outputs: dict[str, Any] | list | None = None,
    type_: str | None = None,
) -> dict:
    """
    Get a dictionary representation of the function node.

    Args:
        func (Callable): The function to be analyzed.
        data_format (str): The format of the output. Options are "semantikon" and
            "ape".

    Returns:
        (dict) A dictionary representation of the function node.
    """

    def regulate_output_keys(
        outputs: dict[str, Any], ref_outputs: dict[str, dict]
    ) -> dict[str, Any]:
        assert len(outputs) == len(ref_outputs), (outputs, ref_outputs)
        ref_keys = list(ref_outputs.keys())
        result = {}
        for key, value in outputs.items():
            try:
                result[ref_keys[int(key)]] = value
            except ValueError:
                result[key] = value
        return result

    new_inputs = parse_input_args(function)
    new_outputs = _get_node_outputs(function)
    if isinstance(outputs, dict):
        assert isinstance(inputs, dict), inputs
        outputs = regulate_output_keys(outputs, new_outputs)
        for key, value in outputs.items():
            new_outputs[key]["value"] = value
        for key, value in inputs.items():
            new_inputs[key]["value"] = value
    if type_ is None:
        type_ = "Function"
    data = {
        "inputs": new_inputs,
        "outputs": new_outputs,
        "function": function,
        "type": type_,
    }
    if hasattr(function, "_semantikon_metadata"):
        data.update(function._semantikon_metadata)
    return data


def to_semantikon_workflow_dict(data: dict) -> dict:
    """
    Convert a workflow dictionary to the Semantikon format.

    Args:
        data (dict): The workflow dictionary to be converted.

    Returns:
        dict: The workflow dictionary in the semantikon format.
    """
    data = copy.deepcopy(data)
    if "nodes" in data:
        for key, node in data["nodes"].items():
            data["nodes"][key] = to_semantikon_workflow_dict(node)
    if "function" in data:
        data.update(
            get_node_dict(
                function=data["function"],
                inputs=data.get("inputs"),
                outputs=data.get("outputs"),
                type_=data.get("type"),
            )
        )
    elif "test" in data:
        data["test"] = get_node_dict(function=data["test"]["function"])
    return data


def get_workflow_dict(func: Callable) -> dict[str, object]:
    """
    Get a dictionary representation of the workflow for a given function.

    Args:
        func (Callable): The function to be analyzed.

    Returns:
        dict: A dictionary representation of the workflow, including inputs,
            outputs, nodes, edges, and label.
    """
    wf = fwf.get_workflow_dict(func, with_function=True)
    return to_semantikon_workflow_dict(wf)


def get_ports(
    func: Callable, separate_return_tuple: bool = True, strict: bool = False
) -> tuple[Inputs, Outputs]:
    type_hints = get_annotated_type_hints(func)
    return_hint = type_hints.pop("return", inspect.Parameter.empty)
    return_labels = get_return_labels(
        func, separate_tuple=separate_return_tuple, strict=strict
    )
    if get_origin(return_hint) is tuple and separate_return_tuple:
        output_annotations = {
            label: meta_to_dict(ann, flatten_metadata=False)
            for label, ann in zip(return_labels, get_args(return_hint))
        }
    else:
        output_annotations = {
            return_labels[0]: meta_to_dict(return_hint, flatten_metadata=False)
        }
    input_annotations = {
        key: meta_to_dict(
            type_hints.get(key, value.annotation), value.default, flatten_metadata=False
        )
        for key, value in inspect.signature(func).parameters.items()
    }
    return (
        Inputs(**{k: Input(label=k, **v) for k, v in input_annotations.items()}),
        Outputs(**{k: Output(label=k, **v) for k, v in output_annotations.items()}),
    )


def get_node(func: Callable, label: str | None = None) -> Function | Workflow:
    metadata_dict = (
        func._semantikon_metadata if hasattr(func, "_semantikon_metadata") else MISSING
    )
    metadata = (
        metadata_dict
        if isinstance(metadata_dict, Missing)
        else CoreMetadata.from_dict(metadata_dict)
    )

    if isinstance(func, fwf.FunctionWithWorkflow):
        return parse_workflow(get_workflow_dict(func), metadata)
    else:
        return parse_function(func, metadata, label=label)


def parse_function(
    func: Callable, metadata: CoreMetadata | Missing, label: str | None = None
) -> Function:
    inputs, outputs = get_ports(func)
    return Function(
        label=func.__name__ if label is None else label,
        inputs=inputs,
        outputs=outputs,
        function=func,
        metadata=metadata,
    )


def _port_from_dictionary(
    io_dictionary: dict[str, object], label: str, port_class: type[PortType]
) -> PortType:
    """
    Take a traditional _semantikon_workflow dictionary's input or output subdictionary
    and nest the metadata (if any) as a dataclass.
    """
    metadata_kwargs = {}
    for field in dataclasses.fields(TypeMetadata):
        if field.name in io_dictionary:
            metadata_kwargs[field.name] = io_dictionary.pop(field.name)
    if len(metadata_kwargs) > 0:
        io_dictionary["metadata"] = TypeMetadata.from_dict(metadata_kwargs)
    io_dictionary["label"] = label
    return port_class.from_dict(io_dictionary)


def _input_from_dictionary(io_dictionary: dict[str, object], label: str) -> Input:
    return _port_from_dictionary(io_dictionary, label, Input)


def _output_from_dictionary(io_dictionary: dict[str, object], label: str) -> Output:
    return _port_from_dictionary(io_dictionary, label, Output)


def parse_workflow(
    semantikon_workflow: dict[str, Any], metadata: CoreMetadata | Missing = MISSING
) -> Workflow:
    label = semantikon_workflow["label"]
    inputs = Inputs(
        **{
            k: _input_from_dictionary(v, label=k)
            for k, v in semantikon_workflow["inputs"].items()
        }
    )
    outputs = Outputs(
        **{
            k: _output_from_dictionary(v, label=k)
            for k, v in semantikon_workflow["outputs"].items()
        }
    )
    nodes = Nodes(
        **{
            k: (
                get_node(v["function"], label=k)
                if v["type"] == "Function"
                else parse_workflow(v)
            )
            for k, v in semantikon_workflow["nodes"].items()
        }
    )
    edges = Edges(**{v: k for k, v in semantikon_workflow["edges"]})
    return Workflow(
        label=label,
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        edges=edges,
        metadata=metadata,
    )


def workflow(func: Callable) -> SemantikonFunctionWithWorkflow:
    """
    Decorator to convert a function into a workflow with metadata.


    Args:
        func (Callable): The function to be converted into a workflow.


    Returns:
        FunctionWithWorkflow: A callable object that includes the original function


    Example:


    >>> def operation(x: float, y: float) -> tuple[float, float]:
    >>>     return x + y, x - y
    >>>
    >>>
    >>> def add(x: float = 2.0, y: float = 1) -> float:
    >>>     return x + y
    >>>
    >>>
    >>> def multiply(x: float, y: float = 5) -> float:
    >>>     return x * y
    >>>
    >>>
    >>> @workflow
    >>> def example_macro(a=10, b=20):
    >>>     c, d = operation(a, b)
    >>>     e = add(c, y=d)
    >>>     f = multiply(e)
    >>>     return f
    >>>
    >>>
    >>> @workflow
    >>> def example_workflow(a=10, b=20):
    >>>     y = example_macro(a, b)
    >>>     z = add(y, b)
    >>>     return z

)
    This example defines a workflow `example_macro`, that includes `operation`,
    `add`, and `multiply`, which is nested inside another workflow
    `example_workflow`. Both workflows can be executed using their `run` method,
    which returns the dictionary representation of the workflow with all the
    intermediate steps and outputs.
    """
    func_with_metadata = SemantikonFunctionWithWorkflow(func)
    return func_with_metadata
