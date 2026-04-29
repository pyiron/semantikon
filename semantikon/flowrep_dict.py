"""
Convert live node trees to the legacy nested-dictionary format.

This module bridges :mod:`flowrep.live` objects (recipe + instance data)
to the ``dict`` structure historically produced by
:func:`flowrep.workflow.get_workflow_dict` in versions <0.3.0
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
import json
import re
from collections.abc import Mapping
from typing import Annotated, Any, cast, get_args, get_origin

import networkx as nx
from flowrep.api import schemas as frs
from pyiron_snippets import retrieve

from semantikon.converter import (
    get_function_dict,
    get_function_metadata,
    meta_to_dict,
)


def live_to_dict(
    node: frs.LiveNode,
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
            :func:`get_function_metadata` dicts (``False``).
        label: Override the inferred label.

    Returns:
        A nested dictionary matching the structure of the legacy
        ``get_workflow_dict`` output.
    """
    if isinstance(node, frs.LiveAtomic):
        return _atomic_to_dict(node, with_io=with_io, with_function=with_function)
    if isinstance(node, frs.LiveWorkflow):
        return _workflow_to_dict(
            node, with_io=with_io, with_function=with_function, label=label
        )
    if isinstance(node, frs.FlowControl):
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
    node: frs.LiveAtomic,
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
    node: frs.LiveWorkflow,
    *,
    with_io: bool,
    with_function: bool,
    label: str | None,
) -> dict[str, Any]:
    recipe = node.recipe
    assert isinstance(recipe, frs.WorkflowNode)

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


def _infer_label(recipe: frs.LiveWorkflowNode) -> str:
    """Best-effort label from a workflow recipe's reference."""
    if recipe.reference is not None:
        return recipe.reference.info.fully_qualified_name.rsplit(".", 1)[-1]
    return ""


# ---------------------------------------------------------------------------
# Edge conversion
# ---------------------------------------------------------------------------


def _workflow_edges(recipe: frs.WorkflowNode) -> list[tuple[str, str]]:
    """Flatten typed edge objects into ``("src", "tgt")`` string tuples."""
    edges: list[tuple[str, str]] = []

    needs_output_sanitization: set[str | None] = {
        # semantikon uses "output" instead of "output_0" for default of one return
        label
        for label, child in recipe.nodes.items()
        if len(child.outputs) == 1 and child.outputs[0] == "output_0"
    }
    if len(recipe.nodes) == 1 and recipe.outputs[0] == "output_0":
        needs_output_sanitization.add(None)  # OutputTargets force .node to be None

    def _sanitize_port_label(handle) -> str:
        if handle.node in needs_output_sanitization:
            return "output"
        return handle.port

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
                f"{source.node}.outputs.{_sanitize_port_label(source)}",
                f"{target.node}.inputs.{target.port}",
            )
        )

    # Child output/passthrough input → workflow output
    for target, source in recipe.output_edges.items():
        if isinstance(source, frs.InputSource):
            edges.append(
                (
                    f"inputs.{source.port}",
                    f"outputs.{_sanitize_port_label(target)}",
                )
            )
        else:
            edges.append(
                (
                    f"{source.node}.outputs.{_sanitize_port_label(source)}",
                    f"outputs.{_sanitize_port_label(target)}",
                )
            )

    return edges


# ---------------------------------------------------------------------------
# Port conversion
# ---------------------------------------------------------------------------


def _port_dict(value, annotation, default=frs.NOT_DATA):
    d = {}
    if not isinstance(value, frs.NotData):
        d["value"] = value
    if annotation is not None:
        d["dtype"] = _unwrap_annotated(annotation)
    if not isinstance(default, frs.NotData):
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
    ports: Mapping[str, frs.InputPort],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, port in ports.items():
        result[name] = _port_dict(port.value, port.annotation, port.default)
    return result


def _output_ports_to_dict(
    ports: Mapping[str, frs.OutputPort],
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
        hash_dict_tmp: dict[str, Any] = {
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
                val = data["value"]
                is_dataclass_instance = dataclasses.is_dataclass(
                    val
                ) and not isinstance(val, type)
                if is_dataclass_instance:
                    hash_dict_tmp["inputs"][inp_name] = dataclasses.asdict(val)
                else:
                    hash_dict_tmp["inputs"][inp_name] = val
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
        elif orig.io is None or orig.arg is None or dest.io is None or dest.arg is None:
            raise ValueError("Origin and/or destination GNodes hit a non-stringy case.")
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
    Return a deep-copied version of the data dictionary with any function
    objects replaced by their serialized metadata.

    Args:
        data (dict[str, Any]): The data dictionary containing nodes and
            functions.

    Returns:
        dict[str, Any]: The modified data dictionary where function objects
            (e.g. in ``"function"`` or ``"test"["function"]`` fields) have
            been replaced by the result of :func:`get_function_metadata`.
    """
    data = copy.deepcopy(data)
    if "nodes" in data:
        for key, node in data["nodes"].items():
            data["nodes"][key] = serialize_functions(node)
    elif "function" in data and not isinstance(data["function"], str):
        data["function"] = get_function_metadata(data["function"])
    return data


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


# ---------------------------------------------------------------------------
# Workflow dict → graph conversion
# ---------------------------------------------------------------------------


class _WorkflowFlattener:
    @staticmethod
    def _remove_us(*args: str) -> str:
        return ".".join(args)

    @staticmethod
    def _dot(*args: str | None) -> str:
        return ".".join(a for a in args if a is not None)

    @classmethod
    def flatten(cls, wf_dict: dict, prefix: str) -> tuple[dict, dict, list[list[str]]]:
        node_dict = cls._serialize_node_metadata(wf_dict, prefix)
        channel_dict = cls._serialize_channels(wf_dict, prefix)
        n, c, e = cls._serialize_children(wf_dict, prefix)
        edge_list = cls._serialize_edges(wf_dict, prefix)
        node_dict.update(n)
        channel_dict.update(c)
        edge_list.extend(e)
        return node_dict, channel_dict, edge_list

    @staticmethod
    def _serialize_node_metadata(wf_dict: dict, prefix: str) -> dict:
        node_dict = {
            prefix: {k: v for k, v in wf_dict.items() if k not in {"nodes", "edges"}}
        }

        assert "function" in wf_dict or wf_dict["type"] != "atomic"

        if "function" in wf_dict:
            meta = get_function_dict(wf_dict["function"])
            meta["identifier"] = ".".join(
                (meta["module"], meta["qualname"], meta["version"])
            )
            node_dict[prefix]["function"] = meta
        return node_dict

    @classmethod
    def _serialize_channels(cls, wf_dict: dict, prefix: str) -> dict:
        channel_dict = {}
        for io_type in ("inputs", "outputs"):
            for pos, (arg, channel) in enumerate(wf_dict.get(io_type, {}).items()):
                label = cls._remove_us(prefix, io_type, arg)
                assert "semantikon_type" not in channel
                if "dtype" in channel:
                    channel.update(meta_to_dict(channel["dtype"]))

                channel_dict[label] = channel | {
                    "semantikon_type": io_type,
                    "position": channel.get("position", pos),
                    "arg": arg,
                }
        return channel_dict

    @classmethod
    def _serialize_children(
        cls, wf_dict: dict, prefix: str
    ) -> tuple[dict, dict, list[list[str]]]:
        node_dict = {}
        channel_dict = {}
        edge_list = []
        for key, node in wf_dict.get("nodes", {}).items():
            child_key = cls._dot(prefix, key)
            n, c, e = cls.flatten(node, child_key)
            node_dict.update(n)
            channel_dict.update(c)
            edge_list.extend(e)
            node_dict[child_key]["parent"] = prefix.replace(".", "-")
        return node_dict, channel_dict, edge_list

    @classmethod
    def _serialize_edges(cls, wf_dict: dict, prefix: str) -> list[list[str]]:
        edge_list = []
        for edge in wf_dict.get("edges", []):
            edge_list.append([cls._remove_us(prefix, a) for a in edge])
        return edge_list


class _WorkflowGraphSerializer:
    """
    Serializes a workflow dictionary into a NetworkX directed graph, where
    nodes represent workflow steps and channels,
    """

    @classmethod
    def serialize(cls, wf_dict: dict) -> nx.DiGraph:
        node_dict, channel_dict, edge_list = _WorkflowFlattener.flatten(
            wf_dict, wf_dict["label"]
        )
        G = nx.DiGraph()

        cls._add_channels(G, channel_dict)
        cls._add_nodes(G, node_dict)
        cls._add_edges(G, edge_list)

        return cls._relabel_graph(G)

    @classmethod
    def _add_channels(cls, G: nx.DiGraph, channel_dict: dict) -> None:
        for key, data in channel_dict.items():
            G.add_node(
                key,
                step=data["semantikon_type"],
                **{k: v for k, v in data.items() if k != "semantikon_type"},
            )

    @classmethod
    def _add_nodes(cls, G: nx.DiGraph, node_dict: dict) -> None:
        for key, data in node_dict.items():
            if "." not in key:
                G.name = key

            G.add_node(
                key,
                step="node",
                **{k: v for k, v in data.items() if k not in {"inputs", "outputs"}},
            )

            for inp in data.get("inputs", {}):
                G.add_edge(f"{key}.inputs.{inp}", key)

            for out in data.get("outputs", {}):
                G.add_edge(key, f"{key}.outputs.{out}")

    @classmethod
    def _add_edges(cls, G: nx.DiGraph, edge_list: list[list[str]]) -> None:
        for edge in edge_list:
            G.add_edge(*edge)

    @staticmethod
    def _relabel_graph(G: nx.DiGraph) -> nx.DiGraph:
        mapping = {n: n.replace(".", "-") for n in G.nodes()}
        return nx.relabel_nodes(G, mapping, copy=True)
