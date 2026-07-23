"""
Convert node data trees to the legacy nested-dictionary format.

This module bridges :mod:`flowrep` retrospective objects (instance data with associated
recipe) to the ``dict`` structure historically produced by
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

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Annotated, Any, cast, get_args, get_origin

import flowrep as fr
import networkx as nx
from pydantic import ValidationError
from pyiron_snippets import retrieve

from semantikon.converter import (
    get_function_dict,
    get_function_metadata,
    meta_to_dict,
    parse_metadata,
)
from semantikon.datastructure import TypeMetadata


def dict_to_nodedata(node: dict[str, Any]) -> fr.schemas.NodeData:
    """Convert a legacy semantikon workflow dict into flowrep node data."""
    try:
        match node["type"]:
            case "atomic":
                return _dict_to_atomic_data(node)
            case "workflow":
                return _dict_to_workflow_data(node)
            case other:
                raise TypeError(f"Unsupported workflow dict node type: {other!r}")
    except ValidationError as exc:
        raise AssertionError(str(exc)) from exc


def nodedata2dict(
    node: fr.schemas.NodeData,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    """Convert a data node to the nested dictionary format.

    Args:
        node: The node data to convert (pre- or post-run).
        label: Override the inferred label.

    Returns:
        A nested dictionary matching the structure of the legacy
        ``get_workflow_dict`` output.
    """
    if isinstance(node, fr.schemas.AtomicData):
        return _atomic_to_dict(node)
    if isinstance(node, fr.schemas.DagData):
        return _workflow_to_dict(node, label=label)
    if isinstance(node, fr.schemas.FlowControlData):
        raise NotImplementedError(
            "FlowControl → dict conversion is not yet implemented.  "
            "The legacy format flattens body-workflow children into the "
            "control-flow dict, which requires unwrapping the body recipe."
        )
    raise TypeError(f"Unsupported data node type: {type(node).__name__}")


def _dict_to_annotation(port: dict[str, Any]) -> Any | None:
    dtype = port.get("dtype")
    metadata = {
        key: value
        for key, value in port.items()
        if key not in {"value", "dtype", "default"}
    }
    if dtype is None and not metadata:
        return None
    if metadata:
        return Annotated[dtype if dtype is not None else Any, metadata]
    return dtype


def _dict_to_input_port(port: dict[str, Any]) -> fr.schemas.InputDataPort:
    return fr.schemas.InputDataPort(
        value=port["value"] if "value" in port else fr.schemas.NOT_DATA,
        annotation=_dict_to_annotation(port),
        default=port["default"] if "default" in port else fr.schemas.NOT_DATA,
    )


def _dict_to_output_port(port: dict[str, Any]) -> fr.schemas.OutputDataPort:
    return fr.schemas.OutputDataPort(
        value=port["value"] if "value" in port else fr.schemas.NOT_DATA,
        annotation=_dict_to_annotation(port),
    )


def _flowrep_recipe_from_callable(function: Any, *, node_type: str):
    if hasattr(function, "flowrep_recipe"):
        return function.flowrep_recipe
    decorator = fr.tools.atomic if node_type == "atomic" else fr.tools.workflow
    return decorator(function).flowrep_recipe


def _dict_to_atomic_recipe(node: dict[str, Any]) -> fr.schemas.AtomicRecipe:
    function = node["function"]
    return _flowrep_recipe_from_callable(function, node_type="atomic")


def _dict_to_recipe(node: dict[str, Any]) -> fr.schemas.RecipeDiscrimination:
    match node["type"]:
        case "atomic":
            return _dict_to_atomic_recipe(node)
        case "workflow":
            return _dict_to_workflow_recipe(node)
        case other:
            raise TypeError(f"Unsupported workflow dict node type: {other!r}")


def _dict_to_atomic_data(node: dict[str, Any]) -> fr.schemas.AtomicData:
    data = fr.schemas.AtomicData.from_recipe(_dict_to_atomic_recipe(node))
    data.function = node["function"]
    data.input_ports = {
        label: _dict_to_input_port(port)
        for label, port in node.get("inputs", {}).items()
    }
    data.output_ports = {
        label: _dict_to_output_port(port)
        for label, port in node.get("outputs", {}).items()
    }
    return data


def _split_endpoint(endpoint: str) -> tuple[str | None, str, str]:
    if endpoint.startswith("inputs.") or endpoint.startswith("outputs."):
        io, port = endpoint.split(".", 1)
        return None, io, port
    parts = endpoint.rsplit(".", 2)
    if len(parts) != 3 or parts[1] not in {"inputs", "outputs"}:
        raise ValueError(f"Malformed workflow edge endpoint: {endpoint!r}")
    return parts[0], parts[1], parts[2]


def _normalize_output_label(label: str, recipe_outputs: list[str]) -> str:
    if label == "output" and recipe_outputs == ["output_0"]:
        return "output_0"
    return label


def _dict_to_workflow_recipe(node: dict[str, Any]) -> fr.schemas.WorkflowRecipe:
    function = node.get("function")
    if function is not None:
        base_recipe = _flowrep_recipe_from_callable(function, node_type="workflow")
        inputs = list(base_recipe.inputs)
        outputs = list(base_recipe.outputs)
        reference = base_recipe.reference
        description = base_recipe.description
    else:
        # Function-less workflows (e.g. pyiron_workflow generic ``Workflow``s) map
        # to reference-free flowrep workflows; take IO from the dict itself.
        inputs = list(node.get("inputs", {}))
        outputs = list(node.get("outputs", {}))
        reference = None
        description = None
    nodes = {
        label: _dict_to_recipe(child_node)
        for label, child_node in node.get("nodes", {}).items()
    }
    input_edges: fr.schemas.InputEdges = {}
    edges: fr.schemas.Edges = {}
    output_edges: fr.schemas.OutputEdges = {}
    for src, tgt in node.get("edges", []):
        src_node, src_io, src_port = _split_endpoint(src)
        tgt_node, tgt_io, tgt_port = _split_endpoint(tgt)
        if (
            src_node is None
            and src_io == "inputs"
            and tgt_node is not None
            and tgt_io == "inputs"
        ):
            input_edges[fr.schemas.TargetHandle(node=tgt_node, port=tgt_port)] = (
                fr.schemas.InputSource(port=src_port)
            )
        elif (
            src_node is not None
            and src_io == "outputs"
            and tgt_node is not None
            and tgt_io == "inputs"
        ):
            src_port = _normalize_output_label(src_port, nodes[src_node].outputs)
            edges[fr.schemas.TargetHandle(node=tgt_node, port=tgt_port)] = (
                fr.schemas.SourceHandle(node=src_node, port=src_port)
            )
        elif (
            tgt_node is None
            and tgt_io == "outputs"
            and src_node is None
            and src_io == "inputs"
        ):
            tgt_port = _normalize_output_label(tgt_port, outputs)
            output_edges[fr.schemas.OutputTarget(port=tgt_port)] = (
                fr.schemas.InputSource(port=src_port)
            )
        elif (
            tgt_node is None
            and tgt_io == "outputs"
            and src_node is not None
            and src_io == "outputs"
        ):
            src_port = _normalize_output_label(src_port, nodes[src_node].outputs)
            tgt_port = _normalize_output_label(tgt_port, outputs)
            output_edges[fr.schemas.OutputTarget(port=tgt_port)] = (
                fr.schemas.SourceHandle(node=src_node, port=src_port)
            )
        else:
            raise ValueError(f"Malformed workflow edge: {src!r} -> {tgt!r}")
    return fr.schemas.WorkflowRecipe(
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        input_edges=input_edges,
        edges=edges,
        output_edges=output_edges,
        reference=reference,
        description=description,
    )


def _dict_to_workflow_data(node: dict[str, Any]) -> fr.schemas.DagData:
    recipe = _dict_to_workflow_recipe(node)
    data = fr.schemas.DagData.from_recipe(recipe)
    data.input_ports = {
        label: _dict_to_input_port(port)
        for label, port in node.get("inputs", {}).items()
    }
    data.output_ports = {
        label: _dict_to_output_port(port)
        for label, port in node.get("outputs", {}).items()
    }
    data.nodes = {
        label: dict_to_nodedata(child_node)
        for label, child_node in node.get("nodes", {}).items()
    }
    return data


# ---------------------------------------------------------------------------
# Atomic
# ---------------------------------------------------------------------------


def _atomic_to_dict(node: fr.schemas.AtomicData) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "atomic"}
    result["function"] = node.function
    result["inputs"] = _input_ports_to_dict(node.input_ports)
    result["outputs"] = _output_ports_to_dict(node.output_ports)
    if hasattr(node.function, "_semantikon_metadata"):
        result.update(node.function._semantikon_metadata)
    return result


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def _workflow_to_dict(node: fr.schemas.DagData, *, label: str | None) -> dict[str, Any]:
    recipe = node.recipe
    assert isinstance(recipe, fr.schemas.WorkflowRecipe)

    result: dict[str, Any] = {
        "type": "workflow",
        "label": label or _infer_label(recipe),
        "nodes": {
            child_label: nodedata2dict(child_node, label=child_label)
            for child_label, child_node in node.nodes.items()
        },
        "edges": _workflow_edges(recipe),
    }

    result["inputs"] = _input_ports_to_dict(node.input_ports)
    result["outputs"] = _output_ports_to_dict(node.output_ports)

    if recipe.reference is not None:
        func = retrieve.import_from_string(recipe.reference.info.fully_qualified_name)
        if hasattr(func, "_semantikon_metadata"):
            result.update(func._semantikon_metadata)
        result["function"] = func

    return result


def _infer_label(recipe: fr.schemas.LiveWorkflowNode) -> str:
    """Best-effort label from a workflow recipe's reference."""
    if recipe.reference is not None:
        return recipe.reference.info.fully_qualified_name.rsplit(".", 1)[-1]
    return ""


# ---------------------------------------------------------------------------
# Edge conversion
# ---------------------------------------------------------------------------


def _workflow_edges(recipe: fr.schemas.WorkflowRecipe) -> list[tuple[str, str]]:
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
        if isinstance(source, fr.schemas.InputSource):
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


def _port_dict(
    value: Any, annotation: Any, default: Any = fr.schemas.NOT_DATA
) -> dict[str, Any]:
    d: dict[str, Any] = {}
    if not isinstance(value, fr.schemas.NotData):
        d["value"] = value
    if type_hint := annotation_to_type_hint(annotation):
        d["dtype"] = type_hint
    if type_metadata := annotation_to_type_metadata(annotation):
        d.update(type_metadata.to_dictionary())
    if not isinstance(default, fr.schemas.NotData):
        d["default"] = default
    return d


def annotation_to_type_hint(annotation: Any) -> Any | None:
    """
    Extract the underlying type hint from a parameter annotation.

    Plain annotations (``int`` in ``def foo(x: int)``) are returned as-is;
    ``Annotated[T, ...]`` is unwrapped to ``T``.

    Args:
        annotation: A parameter annotation, plain or ``Annotated``.

    Returns:
        The underlying type, or ``None`` if ``annotation`` is ``None``.
    """
    if annotation is None:
        return None
    elif get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    else:
        return annotation


def annotation_to_type_metadata(annotation: Any) -> TypeMetadata | None:
    """
    Extract ``TypeMetadata`` from an ``Annotated`` parameter annotation.

    Plain annotations (``int`` in ``def foo(x: int)``) carry no metadata and
    yield ``None``; only ``Annotated[T, ...]`` is parsed.

    Args:
        annotation: A parameter annotation, plain or ``Annotated``.

    Returns:
        The parsed metadata, or ``None`` if ``annotation`` is not ``Annotated``.
    """
    if annotation is None or get_origin(annotation) is not Annotated:
        return None
    return parse_metadata(annotation)


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
    ports: Mapping[str, fr.schemas.InputDataPort],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, port in ports.items():
        result[name] = _port_dict(port.value, port.annotation, port.default)
    return result


def _output_ports_to_dict(
    ports: Mapping[str, fr.schemas.OutputDataPort],
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
