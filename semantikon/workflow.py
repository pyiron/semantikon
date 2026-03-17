import copy
import functools
import keyword
from collections import Counter
from typing import Any, Callable, Iterable, cast

from flowrep import workflow as fwf
from flowrep.models.api import parsers, schemas, wfms

from semantikon import flowrep_dict
from semantikon.converter import (
    get_return_expressions,
    parse_input_args,
    parse_output_args,
)


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


def _validate_label(label: str, func: Callable) -> bool:
    """
    Validate that a label is a valid Python identifier and not a keyword.

    Args:
        label: The label to validate
        func: The function being analyzed (used for error messages)

    Raises:
        ValueError: If the label is not a valid identifier or is a keyword
    """
    if not label.isidentifier() or keyword.iskeyword(label):
        func_name = getattr(func, "__name__", repr(func))
        raise ValueError(
            f"Invalid output label '{label}' for function {func_name}. "
            f"Label must be a valid Python identifier and not a keyword."
        )
    return True


def _get_node_outputs(func: Callable, counts: int | None = None) -> dict[str, dict]:
    output_hints = parse_output_args(
        func, separate_tuple=(counts is None or counts > 1)
    )
    output_vars = get_return_expressions(func)
    if output_vars is None or len(output_vars) == 0:
        return {}
    if (counts is not None and counts == 1) or isinstance(output_vars, str):
        if not isinstance(output_vars, str):
            output_vars = "output"
        label = cast(dict, output_hints).get("label", output_vars)
        assert _validate_label(label, func)
        return {label: cast(dict, output_hints)}
    assert isinstance(output_vars, tuple), output_vars
    assert counts is None or len(output_vars) >= counts, output_vars
    if output_hints == {}:
        return {key: {} for key in output_vars if _validate_label(key, func)}
    else:
        assert counts is None or len(output_hints) >= counts
        result: dict[str, dict] = {}
        for key, hint in zip(output_vars, output_hints):
            label = hint.get("label", key)
            assert _validate_label(label, func)
            if label in result:
                func_name = getattr(func, "__name__", repr(func))
                raise ValueError(
                    f"Duplicate output label '{label}' detected for function "
                    f"{func_name}. Each output must have a unique label. "
                    f"output_vars={output_vars!r}, output_hints={output_hints!r}"
                )
            result[label] = hint
        return result


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
        type_ = "atomic"
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
        counts = _edges_to_output_counts(data["edges"])
        for key, node in data["nodes"].items():
            data["nodes"][key] = to_semantikon_workflow_dict(
                node, output_counts=counts.get(key)
            )
    if "edges" in data:
        data["edges"] = _flowrep_to_semantikon_edges(data)
    return data


def workflow(func: Callable) -> Callable:
    func = parsers.workflow(func)
    # Expose new dictionary getter
    func.get_semantikon_dict = functools.partial(  # type: ignore[attr-defined]
        _get_semantikon_dict, func
    )
    # Override flowrep bound run method (always with_function)
    func.run = functools.partial(run_workflow_dict, func)  # type: ignore[attr-defined]
    return func


workflow.__doc__ = fwf.workflow.__doc__


def _get_semantikon_dict(workflow_func):
    # Assumes *workflow_func* is already a flowrep workflow recipe holder
    return to_semantikon_workflow_dict(
        flowrep_dict.live_to_dict(
            schemas.Workflow.from_recipe(workflow_func.flowrep_recipe),
            with_io=True,
            with_function=True,
        )
    )


def run_workflow_dict(func, **kwargs) -> dict[str, Any]:
    executed = wfms.run_recipe(func.flowrep_recipe, **kwargs)
    wf_dict = flowrep_dict.live_to_dict(executed, with_io=True, with_function=True)
    return to_semantikon_workflow_dict(wf_dict)
