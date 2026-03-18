"""
Convert live node trees to the legacy nested-dictionary format.

This module bridges :mod:`flowrep.models.live` objects (recipe + instance data)
to the ``dict`` structure historically produced by :func:`flowrep.workflow.get_workflow_dict`
and consumed by downstream packages such as *semantikon*.

Edge string format
------------------
``"inputs.{port}"``
    Workflow-level input port.
``"outputs.{port}"``
    Workflow-level output port.
``"{child_label}.inputs.{port}"``
    Child node input port.
``"{child_label}.outputs.{port}"``
    Child node output port.

.. note::

   Port names are taken directly from the flowrep recipe (e.g. ``"output_0"``,
   ``"quotient"``).  The legacy ``workflow.py`` used ``"output"`` for
   single-output atomics and numeric indices for multi-output; the new naming
   is self-consistent within edges and should be transparent to any consumer
   that follows edges rather than hard-coding port names.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import hashlib
import importlib
import inspect
import json
import re
import textwrap
from collections.abc import Callable, Mapping
from typing import Annotated, Any, cast, get_args, get_origin

import networkx as nx
from pyiron_snippets import retrieve

from flowrep.models import edge_models, live
from flowrep.models.nodes import workflow_model


def live_to_dict(
    node: live.LiveNode,
    *,
    with_io: bool = False,
    with_function: bool = False,
    label: str | None = None,
) -> dict[str, Any]:
    """Convert a live node to the nested dictionary format.

    Args:
        node: The live node to convert (pre- or post-run).
        with_io: Include ``"inputs"`` / ``"outputs"`` port dictionaries.
        with_function: Store raw callables (``True``) or
            :func:`tools.get_function_metadata` dicts (``False``).
        label: Override the inferred label.

    Returns:
        A nested dictionary matching the structure of the legacy
        ``get_workflow_dict`` output.
    """
    if isinstance(node, live.Atomic):
        return _atomic_to_dict(node, with_io=with_io, with_function=with_function)
    if isinstance(node, live.Workflow):
        return _workflow_to_dict(
            node, with_io=with_io, with_function=with_function, label=label
        )
    if isinstance(node, live.FlowControl):
        raise NotImplementedError(
            "FlowControl → dict conversion is not yet implemented.  "
            "The legacy format flattens body-workflow children into the "
            "control-flow dict, which requires unwrapping the body recipe."
        )
    raise TypeError(f"Unsupported live node type: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Atomic
# ---------------------------------------------------------------------------


def _atomic_to_dict(
    node: live.Atomic,
    *,
    with_io: bool,
    with_function: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "atomic"}
    result["function"] = (
        node.function if with_function else get_function_metadata(node.function)
    )
    if with_io:
        result["inputs"] = _input_ports_to_dict(node.input_ports)
        result["outputs"] = _output_ports_to_dict(node.output_ports)
    return result


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def _workflow_to_dict(
    node: live.Workflow,
    *,
    with_io: bool,
    with_function: bool,
    label: str | None,
) -> dict[str, Any]:
    recipe = node.recipe
    assert isinstance(recipe, workflow_model.WorkflowNode)

    result: dict[str, Any] = {
        "type": "workflow",
        "label": label or _infer_label(recipe),
        "nodes": {
            child_label: live_to_dict(
                child_node,
                with_io=with_io,
                with_function=with_function,
                label=child_label,
            )
            for child_label, child_node in node.nodes.items()
        },
        "edges": _workflow_edges(recipe),
    }

    if with_io:
        result["inputs"] = _input_ports_to_dict(node.input_ports)
        result["outputs"] = _output_ports_to_dict(node.output_ports)

    if with_function and recipe.reference is not None:
        result["function"] = retrieve.import_from_string(
            recipe.reference.info.fully_qualified_name
        )

    return result


def _infer_label(recipe: workflow_model.WorkflowNode) -> str:
    """Best-effort label from a workflow recipe's reference."""
    if recipe.reference is not None:
        return recipe.reference.info.fully_qualified_name.rsplit(".", 1)[-1]
    return ""


# ---------------------------------------------------------------------------
# Edge conversion
# ---------------------------------------------------------------------------


def _workflow_edges(recipe: workflow_model.WorkflowNode) -> list[tuple[str, str]]:
    """Flatten typed edge objects into ``("src", "tgt")`` string tuples."""
    edges: list[tuple[str, str]] = []

    needs_output_sanitization = {
        # semantikon uses "output" instead of "outputs_0" for default of one return
        label
        for label, child in recipe.nodes.items()
        if len(child.outputs) == 1 and child.outputs[0] == "output_0"
    }

    # Workflow input → child input
    for target, source in recipe.input_edges.items():
        edges.append(
            (
                f"inputs.{source.port}",
                f"{target.node}.inputs.{target.port}",
            )
        )

    # Sibling output → sibling input
    for target, source in recipe.edges.items():
        edges.append(
            (
                f"{source.node}.outputs.{'output' if source.node in needs_output_sanitization else source.port}",
                f"{target.node}.inputs.{target.port}",
            )
        )

    # Child output → workflow output  (or passthrough from workflow input)
    for target, source in recipe.output_edges.items():
        if isinstance(source, edge_models.InputSource):
            edges.append(
                (
                    f"inputs.{source.port}",
                    f"outputs.{'output' if target.node in needs_output_sanitization else target.port}",
                )
            )
        else:
            edges.append(
                (
                    f"{source.node}.outputs.{'output' if source.node in needs_output_sanitization else source.port}",
                    f"outputs.{target.port}",
                )
            )

    return edges


# ---------------------------------------------------------------------------
# Port conversion
# ---------------------------------------------------------------------------


def _port_dict(value, annotation, default=live.NOT_DATA):
    d = {}
    if not isinstance(value, live.NotData):
        d["value"] = value
    if annotation is not None:
        d["dtype"] = _unwrap_annotated(annotation)
    if not isinstance(default, live.NotData):
        d["default"] = default
    return d


def _unwrap_annotated(annotation: Any) -> Any:
    """
    Strip ``Annotated`` wrappers, returning just the base type.

    Semantic metadata (``uri``, ``units``, ``triples``, …) embedded in
    ``Annotated`` extras belongs to *semantikon*, not to the flowrep dict.
    Storing the full ``Annotated`` type as ``"dtype"`` would cause
    ``meta_to_dict`` to double-extract that metadata in a wrong context.
    """
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def _input_ports_to_dict(
    ports: Mapping[str, live.InputPort],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, port in ports.items():
        result[name] = _port_dict(port.value, port.annotation, port.default)
    return result


def _output_ports_to_dict(
    ports: Mapping[str, live.OutputPort],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if len(ports) == 1 and next(iter(ports.keys())) == "output_0":
        port = next(iter(ports.values()))
        result["output"] = _port_dict(port.value, port.annotation)
    else:
        for name, port in ports.items():
            result[name] = _port_dict(port.value, port.annotation)
    return result


# ---------------------------------------------------------------------------
# Consumers
# ---------------------------------------------------------------------------


def get_hashed_node_dict(workflow_dict: dict[str, dict]) -> dict[str, Any]:
    G = _get_workflow_graph(workflow_dict)
    hash_dict = {}

    for node in list(nx.topological_sort(G)):
        break_flag = False
        if "@" in node:
            for term in ["hash", "value"]:
                if "hash" not in G.nodes[node]:
                    for pre in G.predecessors(node):
                        if term in G.nodes[pre]:
                            G.nodes[node][term] = G.nodes[pre][term]
                            continue
            continue
        hash_dict_tmp = {
            "inputs": {},
            "outputs": [
                G.nodes[out].get("label", out.split("@")[-1])
                for out in G.successors(node)
            ],
            "node": get_function_metadata(G.nodes[node]["function"]),
        }
        hash_dict_tmp["node"]["connected_inputs"] = []
        for inp in G.predecessors(node):
            data = G.nodes[inp]
            inp_name = inp.split("@")[-1]
            if "hash" in data:
                hash_dict_tmp["inputs"][inp_name] = data["hash"]
                hash_dict_tmp["node"]["connected_inputs"].append(inp_name)
            elif "value" in data:
                if dataclasses.is_dataclass(data["value"]):
                    hash_dict_tmp["inputs"][inp_name] = dataclasses.asdict(
                        data["value"]
                    )
                else:
                    hash_dict_tmp["inputs"][inp_name] = data["value"]
            else:
                break_flag = True
        if break_flag:
            continue
        h = hashlib.sha256(
            json.dumps(hash_dict_tmp, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for out in G.successors(node):
            G.nodes[out]["hash"] = (
                h + "@" + G.nodes[out].get("label", out.split("@")[-1])
            )
        hash_dict_tmp["hash"] = h
        hash_dict[node.replace("/", ".")] = hash_dict_tmp
    return hash_dict


def _get_workflow_graph(workflow_dict: dict[str, Any]) -> nx.DiGraph:
    """
    Convert a workflow dictionary into a directed graph representation.

    Args:
        workflow_dict (dict[str, Any]): The dictionary representation of the
            workflow.

    Returns:
        nx.DiGraph: A directed graph representing the workflow.
    """

    def _to_gnode(node: str) -> str:
        node_split = node.rsplit(".", 2)
        if len(node_split) < 2:
            return node
        if len(node_split) == 2:
            assert node_split[0] in [
                "inputs",
                "outputs",
            ], f"Node {node} is not correctly formatted"
            return f"{node_split[0]}@{node_split[1]}"
        assert node_split[1] in [
            "inputs",
            "outputs",
        ], f"Node {node} is not correctly formatted"
        return f"{node_split[0]}:{node_split[1]}@{node_split[2]}"

    G = nx.DiGraph()
    for inp, data in workflow_dict.get("inputs", {}).items():
        G.add_node(f"inputs@{inp}", step="input", **data)
    for out, data in workflow_dict.get("outputs", {}).items():
        G.add_node(f"outputs@{out}", step="output", **data)

    nodes_to_delete = []
    for key, node in workflow_dict["nodes"].items():
        assert node["type"] in ["atomic", "workflow"]
        if node["type"] == "workflow":
            child_G = _get_workflow_graph(node)
            for child_key in list(child_G.graph.keys()):
                new_key = f"{key}/{child_key}" if child_key != "" else key
                child_G.graph[new_key] = child_G.graph[child_key]
            mapping = {}
            for n in child_G.nodes():
                if n.startswith("inputs@") or n.startswith("outputs@"):
                    mapping[n] = key + ":" + n
                else:
                    mapping[n] = key + "/" + n
            G = nx.union(nx.relabel_nodes(child_G, mapping), G)
            nodes_to_delete.append(key)
        else:
            G.add_node(
                key,
                step="node",
                **{k: v for k, v in node.items() if k not in ["inputs", "outputs"]},
            )
        for ii, (inp, data) in enumerate(node.get("inputs", {}).items()):
            G.add_node(f"{key}:inputs@{inp}", step="input", **({"position": ii} | data))
        for ii, (out, data) in enumerate(node.get("outputs", {}).items()):
            G.add_node(
                f"{key}:outputs@{out}", step="output", **({"position": ii} | data)
            )
    for edge in _get_missing_edges(cast(list[tuple[str, str]], workflow_dict["edges"])):
        G.add_edge(_to_gnode(edge[0]), _to_gnode(edge[1]))
    for node in nodes_to_delete:
        G.remove_node(node)
    for node in G.nodes():
        if "@" not in node:
            continue
        io = node.split(":")[-1].split("@")[0]
        assert io in ["inputs", "outputs"], f"Node {node} is not correctly formatted"
        G.nodes[node]["step"] = {"inputs": "input", "outputs": "output"}[io]
    G.graph[""] = {
        key: value
        for key, value in workflow_dict.items()
        if key not in ["nodes", "edges", "inputs", "outputs"]
    }

    return G


def _get_missing_edges(edge_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Insert processes into the data edges. Take the following workflow:

    >>> y = f(x=x)
    >>> z = g(y=y)

    The data flow is

    - f.inputs.x -> f.outputs.y
    - f.outputs.y -> g.inputs.y
    - g.inputs.y -> g.outputs.z

    `_get_missing_edges` adds the processes:

    - f.inputs.x -> f
    - f -> f.outputs.y
    - f.outputs.y -> g.inputs.y
    - g.inputs.y -> g
    - g -> g.outputs.z
    """
    extra_edges = []
    for edge in edge_list:
        for tag in edge:
            if len(tag.split(".")) < 3:
                continue
            if tag.split(".")[1] == "inputs":
                new_edge = (tag, tag.split(".")[0])
            elif tag.split(".")[1] == "outputs":
                new_edge = (tag.split(".")[0], tag)
            if new_edge not in extra_edges:
                extra_edges.append(new_edge)
    return edge_list + extra_edges


def _simple_run(G: nx.DiGraph) -> nx.DiGraph:
    for node in nx.topological_sort(G):
        data = G.nodes[node]
        if data["step"] == "node":
            if all("value" in G.nodes[succ] for succ in G.successors(node)):
                continue
            kwargs = {}
            for inp in G.predecessors(node):
                kwargs[inp.split("@")[-1]] = G.nodes[inp]["value"]
            outputs = data["function"](**kwargs)
            successors = list(G.successors(node))
            if len(successors) == 1:
                G.nodes[successors[0]]["value"] = outputs
            else:
                for succ in successors:
                    G.nodes[succ]["value"] = outputs[
                        int(G.nodes[succ].get("position", succ.split("@")[-1]))
                    ]
            continue
        if "default" in data and "value" not in data:
            data["value"] = data["default"]
        if G.in_degree(node) == 0 and "value" not in data:
            raise ValueError(f"Input values not given for {node}")
        assert "value" in data
        for succ in G.successors(node):
            if G.nodes[succ].get("type") != "atomic":
                G.nodes[succ]["value"] = data["value"]
    return G


class GNode:
    def __init__(self, key: str):
        self.key = key

    @property
    def node(self) -> str | None:
        if "@" in self.key and ":" not in self.key:
            return None
        return self.key.split(":")[0]

    @property
    def node_list(self) -> list[str]:
        if self.node is None:
            return []
        return self.node.split("/")

    @property
    def io(self) -> str | None:
        arg = re.search(r":(inputs|outputs)@", self.key)
        if arg is not None:
            return arg.group(1)
        arg = re.search(r"(inputs|outputs)@", self.key)
        if arg is not None:
            return arg.group(1)
        return None

    @property
    def arg(self) -> str | None:
        return self.key.split("@")[-1] if "@" in self.key else None

    @property
    def is_io(self) -> bool:
        return self.io is not None


def _graph_to_wf_dict(G: nx.DiGraph) -> dict:
    """
    Convert a directed graph representation of a workflow into a workflow
    dictionary.

    Args:
        G (nx.DiGraph): A directed graph representing the workflow.

    Returns:
        dict: The dictionary representation of the workflow.
    """
    wf_dict = dict_to_recursive_dd({})

    for node, metadata in list(G.nodes.data()):
        gn = GNode(node)
        d = wf_dict
        for n in gn.node_list:
            d = d["nodes"][n]
        if gn.is_io:
            d[gn.io][gn.arg] = {
                key: value for key, value in metadata.items() if key != "step"
            }
        else:
            d.update({key: value for key, value in metadata.items() if key != "step"})

    for edge in G.edges:
        orig = GNode(edge[0])
        dest = GNode(edge[1])
        if not orig.is_io or not dest.is_io:
            continue
        if len(orig.node_list) == len(dest.node_list):
            nodes = orig.node_list[:-1]
            if len(orig.node_list) == 0:
                edge = (
                    ".".join([orig.io, orig.arg]),
                    ".".join([dest.io, dest.arg]),
                )
            else:
                edge = (
                    ".".join([orig.node_list[-1], orig.io, orig.arg]),
                    ".".join([dest.node_list[-1], dest.io, dest.arg]),
                )
        elif len(orig.node_list) > len(dest.node_list):
            nodes = dest.node_list
            edge = (
                ".".join([orig.node_list[-1], orig.io, orig.arg]),
                ".".join([dest.io, dest.arg]),
            )
        else:
            nodes = orig.node_list
            edge = (
                ".".join([orig.io, orig.arg]),
                ".".join([dest.node_list[-1], dest.io, dest.arg]),
            )
        d = wf_dict
        for node in nodes:
            d = d["nodes"][node]
        if not isinstance(d["edges"], list):
            d["edges"] = []
        d["edges"].append(edge)
    for key, value in G.graph.items():
        d = wf_dict
        if key != "":
            for n in key.split("/"):
                d = d["nodes"][n]
        for k, v in value.items():
            d[k] = v
    return recursive_dd_to_dict(wf_dict)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def serialize_functions(data: dict[str, Any]) -> dict[str, Any]:
    """
    Separate functions from the data dictionary and store them in a function
    dictionary. The functions inside the data dictionary will be replaced by
    their name (which would for example make it easier to hash it)

    Args:
        data (dict[str, Any]): The data dictionary containing nodes and
            functions.

    Returns:
        tuple: A tuple containing the modified data dictionary and the
            function dictionary.
    """
    data = copy.deepcopy(data)
    if "nodes" in data:
        for key, node in data["nodes"].items():
            data["nodes"][key] = serialize_functions(node)
    elif "function" in data and not isinstance(data["function"], str):
        data["function"] = get_function_metadata(data["function"])
    if "test" in data and not isinstance(data["test"]["function"], str):
        data["test"]["function"] = get_function_metadata(data["test"]["function"])
    return data


def _get_version_from_module(module_name: str) -> str:
    base_module_name = module_name.split(".")[0]
    base_module = importlib.import_module(base_module_name)
    return getattr(base_module, "__version__", "not_defined")


def get_function_metadata(
    cls: Callable | dict[str, str], full_metadata: bool = False
) -> dict[str, str]:
    """
    Get metadata for a given function or class.

    Args:
        cls (Callable | dict[str, str]): The function or class to get metadata for.
        full_metadata (bool): Whether to include full metadata including hash,
            docstring, and name.

    Returns:
        dict[str, str]: A dictionary containing the metadata of the function or class.
    """
    if isinstance(cls, dict) and "module" in cls and "qualname" in cls:
        return cls
    data = {
        "module": cls.__module__,
        "qualname": cls.__qualname__,
    }

    data["version"] = _get_version_from_module(data["module"])
    if not full_metadata:
        return data
    data["hash"] = hash_function(cls)
    data["docstring"] = cls.__doc__ or ""
    data["name"] = cls.__name__
    return data


def hash_function(fn: Callable) -> str:
    """
    Hash a function based on its source code or signature.

    For regular functions, the hash is based on the dedented source code.
    For other callables (built-ins, methods, etc.), the hash is based on
    the module, qualified name, and signature. If source code is unavailable
    for a function, falls back to signature-based hashing.

    Args:
        fn (Callable): The function to hash.

    Returns:
        str: A stable hash string in the format "function_name:hash_hex".
    """

    if inspect.isfunction(fn):
        try:
            source_code = inspect.getsource(fn)
            source_code = textwrap.dedent(
                source_code.replace("\r\n", "\n").replace("\r", "\n")
            )
        except (OSError, TypeError):
            # Fall back to signature for functions where source is unavailable
            source_code = f"{fn.__module__}:{fn.__qualname__}:{inspect.signature(fn)}"
    else:
        source_code = f"{fn.__module__}:{fn.__qualname__}:{inspect.signature(fn)}"
    source_code_hash = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    name = getattr(fn, "__name__", "unknown")
    return name + ":" + source_code_hash


def recursive_defaultdict() -> collections.defaultdict:
    """
    Create a recursively nested collections.defaultdict.

    Example:
    >>> d = recursive_defaultdict()
    >>> d['a']['b']['c'] = 1
    >>> print(d)

    Output: 1
    """
    return collections.defaultdict(recursive_defaultdict)


def dict_to_recursive_dd(d: dict | collections.defaultdict) -> collections.defaultdict:
    """Convert a regular dict to a recursively nested collections.defaultdict."""
    if isinstance(d, dict) and not isinstance(d, collections.defaultdict):
        return collections.defaultdict(
            recursive_defaultdict, {k: dict_to_recursive_dd(v) for k, v in d.items()}
        )
    return d


def recursive_dd_to_dict(d: dict | collections.defaultdict) -> dict:
    """Convert a recursively nested collections.defaultdict to a regular dict."""
    if isinstance(d, collections.defaultdict):
        return {k: recursive_dd_to_dict(v) for k, v in d.items()}
    return d
