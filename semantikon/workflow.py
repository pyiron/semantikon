import ast
import builtins
import copy
import dataclasses
import inspect
import textwrap
from collections import Counter, deque
from functools import cached_property, update_wrapper
from typing import Any, Callable, Generic, Iterable, TypeVar, cast, get_args, get_origin

import networkx as nx
from flowrep import workflow as fwf
from networkx.algorithms.dag import topological_sort

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


class FunctionWithWorkflow(fwf.FunctionWithWorkflow):
    def run(self, *args, **kwargs) -> dict[str, Any]:
        return to_semantikon_workflow_dict(
            super().run(*args, with_function=True, **kwargs)
        )

    def serialize_workflow(self) -> dict:
        wf_dict = self._serialize_workflow(with_function=True, with_io=True)
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


def _edges_to_output_counts(edges: Iterable[tuple[str, str]]) -> dict[str, int]:
    """
    Get a count of outputs from edges.

    Args:
        edges (Iterable[tuple[str, str]]): An iterable of edges.

    Returns:
        dict[str, int]: A dictionary with output names as keys and their counts as values.

    Example:
        >>> edges = [
        ...     ("node1.outputs.0", "node2.inputs.input1"),
        ...     ("node1.outputs.1", "node3.inputs.input1"),
        ...     ("node2.outputs.output", "node4.inputs.input1")
        ... ]
        >>> _edges_to_output_counts(edges)
        {'node1': 2, 'node2': 1}
    """
    counts = [edge[0].split(".outputs.")[0] for edge in edges if ".outputs." in edge[0]]
    return dict(Counter(counts))


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
    assert counts is None or len(output_vars) >= counts, output_vars
    if output_hints == {}:
        return {key: {} for key in output_vars}
    else:
        assert counts is None or len(output_hints) >= counts
        return {key: hint for key, hint in zip(output_vars, output_hints)}


def _get_node_dict(
    function: Callable,
    inputs: dict[str, dict] | None = None,
    outputs: dict[str, Any] | None = None,
    output_counts: int | None = None,
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
        result = ref_outputs.copy()
        for key, value in outputs.items():
            if len(ref_keys) == 1 and key == "output":
                result[ref_keys[0]].update(value)
                continue
            try:
                result[ref_keys[int(key)]].update(value)
            except ValueError:
                result[key].update(value)
        return result

    new_inputs = parse_input_args(function)
    if inputs is not None:
        for key, value in inputs.items():
            if key in new_inputs:
                new_inputs[key].update(value)
    new_outputs = _get_node_outputs(function, counts=output_counts)
    if outputs is not None:
        new_outputs = regulate_output_keys(outputs, new_outputs)
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


def _flowrep_to_semantikon_edges(wf: dict) -> list[tuple[str, str]]:
    output_tags = {
        key: list(node["outputs"].keys()) for key, node in wf["nodes"].items()
    }
    edges = []
    for edge in wf["edges"]:
        if ".outputs." not in edge[0]:
            edges.append(edge)
        else:
            f, _, port = edge[0].split(".")
            if port == "output":
                assert len(output_tags[f]) == 1
                port = output_tags[f][0]
            else:
                try:
                    port = output_tags[f][int(port)]
                except ValueError:
                    pass
            edges.append((f"{f}.outputs.{port}", edge[1]))
    return edges


def to_semantikon_workflow_dict(data: dict, output_counts: int | None = None) -> dict:
    """
    Convert a workflow dictionary to the Semantikon format.

    Args:
        data (dict): The workflow dictionary to be converted.

    Returns:
        dict: The workflow dictionary in the semantikon format.
    """
    data = copy.deepcopy(data)
    if "function" in data:
        data.update(
            _get_node_dict(
                function=data["function"],
                inputs=data.get("inputs"),
                outputs=data.get("outputs"),
                output_counts=output_counts,
                type_=data["type"],
            )
        )
    elif "test" in data:
        data["test"] = _get_node_dict(function=data["test"]["function"])
    if "nodes" in data:
        assert "edges" in data, data
        output_counts = _edges_to_output_counts(data["edges"])
        for key, node in data["nodes"].items():
            data["nodes"][key] = to_semantikon_workflow_dict(
                node, output_counts=output_counts.get(key)
            )
    if "edges" in data:
        data["edges"] = _flowrep_to_semantikon_edges(data)
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
    wf = fwf.get_workflow_dict(func, with_function=True, with_io=True)
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
    Take a traditional semantikon workflow dictionary's input or output subdictionary
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


def workflow(func: Callable) -> FunctionWithWorkflow:
    func_with_metadata = FunctionWithWorkflow(func)
    return func_with_metadata


workflow.__doc__ = fwf.workflow.__doc__
