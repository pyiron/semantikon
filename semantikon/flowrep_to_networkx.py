from __future__ import annotations

import copy
import json
import unicodedata
import warnings
from abc import ABC
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import cached_property
from hashlib import sha256
from typing import Any

import flowrep as fr
import networkx as nx
from pyiron_snippets import retrieve
from rdflib import BNode
from rdflib.term import IdentifiedNode

from semantikon.converter import get_function_dict
from semantikon.flowrep_dict import (
    annotation_to_type_hint,
    annotation_to_type_metadata,
    dict_to_nodedata,
)


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    parent: Node | None = None

    def __str__(self) -> str:
        if self.parent:
            return f"{self.parent}-{self.name}"
        else:
            return self.name

    def __radd__(self, other: str) -> str:
        return other + str(self)


@dataclass(frozen=True, slots=True)
class IO(ABC):
    node: Node
    port: str

    def __radd__(self, other: str) -> str:
        return other + str(self)


@dataclass(frozen=True, slots=True)
class Input(IO):
    def __str__(self) -> str:
        return f"{self.node}-inputs-{self.port}"


@dataclass(frozen=True, slots=True)
class Output(IO):
    def __str__(self) -> str:
        return f"{self.node}-outputs-{self.port}"


@dataclass(frozen=True, slots=True)
class TNodeData:
    type: str
    identifier: str | None = None
    label: str | None = None
    function: dict | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in {"atomic", "constant", "workflow"}:
            raise ValueError("type must be either 'atomic', 'constant', or 'workflow'")

    def to_attrs(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "type": self.type,
        }
        if self.identifier is not None:
            attrs["identifier"] = self.identifier
        if self.label is not None:
            attrs["label"] = self.label
        if self.function is not None:
            attrs["function"] = self.function
        attrs.update(self.extras)
        return attrs


@dataclass(frozen=True, slots=True)
class TIOData:
    position: int
    dtype: Any | None = None
    value: Any | None = None
    has_value: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_attrs(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "position": self.position,
        }
        if self.dtype is not None:
            attrs["dtype"] = self.dtype
        if self.has_value:
            attrs["value"] = self.value
        attrs.update(self.metadata)
        return attrs


@dataclass(frozen=True, slots=True)
class TInputData(TIOData):
    default: Any | None = None
    has_default: bool = False

    def to_attrs(self) -> dict[str, Any]:
        attrs = TIOData.to_attrs(self)
        if self.has_default:
            attrs["default"] = self.default
        return attrs


@dataclass(frozen=True, slots=True)
class TOutputData(TIOData):
    pass


class SemantikonDiGraph(nx.DiGraph):
    """Workflow graph with deterministic namespace fragments.

    The graph only stores suffix fragments (e.g. ``abc123_``) for type- and
    assertion-level identifiers. Ontology-specific base namespaces are applied
    later by the ontology serialization layer.
    """

    def _validate_semantikon_attrs(
        self, step: IO | Node, attrs: dict[str, Any]
    ) -> dict[str, Any]:

        if isinstance(step, Node):
            known = {"type", "identifier", "label", "function"}
            node_meta = TNodeData(
                type=attrs["type"],
                identifier=attrs.get("identifier"),
                label=attrs.get("label"),
                function=attrs.get("function"),
                extras={k: v for k, v in attrs.items() if k not in known},
            )
            return node_meta.to_attrs()

        if isinstance(step, Input):
            known = {"position", "dtype", "value", "default"}
            input_meta = TInputData(
                position=attrs["position"],
                dtype=attrs.get("dtype"),
                value=attrs.get("value"),
                has_value="value" in attrs,
                default=attrs.get("default"),
                has_default="default" in attrs,
                metadata={k: v for k, v in attrs.items() if k not in known},
            )
            return input_meta.to_attrs()

        if isinstance(step, Output):
            known = {"position", "dtype", "value"}
            output_meta = TOutputData(
                position=attrs["position"],
                dtype=attrs.get("dtype"),
                value=attrs.get("value"),
                has_value="value" in attrs,
                metadata={k: v for k, v in attrs.items() if k not in known},
            )
            return output_meta.to_attrs()

        raise TypeError(f"Unknown step type: {type(step)}")

    def add_node(self, node_for_adding, **attr):  # type: ignore[override]
        assert isinstance(node_for_adding, (Node, IO))
        normalized_attr = self._validate_semantikon_attrs(node_for_adding, attr)
        super().add_node(node_for_adding, **normalized_attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        assert all(isinstance(n, (Node, IO, tuple)) for n in nodes_for_adding)
        for n in nodes_for_adding:
            if isinstance(n, tuple):
                node, node_attr = n
                normalized_attr = self._validate_semantikon_attrs(
                    node, attr | node_attr
                )
                super().add_node(node, **normalized_attr)
            else:
                normalized_attr = self._validate_semantikon_attrs(n, attr)
                super().add_node(n, **normalized_attr)

    @cached_property
    def t_ns(self) -> str:
        """Type-level namespace fragment for this graph."""
        h = (
            "W" + _get_graph_hash(self, with_global_inputs=False)[:8]
            if self.graph["prefix"] is None
            else self.graph["prefix"]
        )
        return h + "_"

    @cached_property
    def a_ns(self) -> str:
        """Assertion-level namespace fragment for this graph."""
        h = _get_graph_hash(self, with_global_inputs=True)
        return h + "_"

    def _get_data_node(self, io: IO) -> str:
        while True:
            candidate = [c for c in self.predecessors(io) if isinstance(c, IO)]
            assert len(candidate) <= 1
            if len(candidate) == 0:
                return f"{io}_data"
            io = candidate[0]

    def append_hash(
        self,
        node: str,
        hash_value: str,
        label: str | None = None,
    ):
        """
        Propagates a hash value through the descendants of a given node in a
        directed graph.

        This function iteratively traverses the descendants of the graph and
        appends a hash value to each descendant node. The hash value is
        updated based on the label of each node. Optionally, it can remove
        specific data (e.g., "value") from the nodes.

        Parameters:
            node (str): The starting node from which the hash propagation begins.
            hash_value (str): The initial hash value to propagate through the
                descendants.
            label (str | None, optional): A label to use for hash computation.
                If not provided, the label is derived from the node's data
                (e.g., "label"). Defaults to None.

        Notes:
            - The function uses an iterative approach to avoid recursion,
                making it suitable for graphs with deep hierarchies.
            - The hash value for each node is updated in the format:
                `parent_hash@child_label`.
        """
        stack = [(node, hash_value, label)]

        while stack:
            current_node, current_hash, current_label = stack.pop()

            for child in self.successors(current_node):
                if isinstance(child, Node):
                    continue

                child_label = current_label
                if child_label is None:
                    child_label = self.nodes[child].get("label", child.port)

                self.nodes[child]["hash"] = current_hash + f"@{child_label}"
                stack.append((child, current_hash, child_label))

    def get_hash_dict(self) -> dict[str, str]:
        """
        Get a dictionary mapping node names to their hash values.

        Returns:
            dict[str, str]: A dictionary where keys are node names and values
                are their corresponding hash values.
        """
        hash_dict = {}
        for _, data in self.nodes.data():
            if "hash" in data and "value" in data:
                hash_dict[data["hash"]] = data["value"]
        return hash_dict


def _infer_workflow_label(recipe: fr.schemas.WorkflowRecipe) -> str:
    if recipe.reference is None:
        return ""
    return recipe.reference.info.fully_qualified_name.rsplit(".", 1)[-1]


def _output_port_label(port: str, outputs: list[str]) -> str:
    if port == "output_0" and len(outputs) == 1 and outputs[0] == "output_0":
        return "output"
    return port


def _workflow_to_networkx(
    workflow: fr.schemas.DagData,
    *,
    prefix: str | None = None,
) -> SemantikonDiGraph:
    root_label = _infer_workflow_label(workflow.recipe)
    G = SemantikonDiGraph(prefix=prefix)
    G.name = root_label

    def _add_node(
        node_data: fr.schemas.NodeData,
        node_name: Node,
        *,
        parent_name: Node | None = None,
        workflow_label: Node | None = None,
    ):
        metadata: dict[str, Any] = {}
        function = None
        if isinstance(node_data, fr.schemas.AtomicData):
            metadata["type"] = "atomic"
            function = node_data.function
        elif isinstance(node_data, fr.schemas.ConstantData):
            metadata["type"] = "constant"
        else:
            metadata["type"] = "workflow"
            if workflow_label is not None:
                metadata["label"] = str(workflow_label)
            if node_data.recipe.reference is not None:
                function = retrieve.import_from_string(
                    node_data.recipe.reference.info.fully_qualified_name
                )

        if function is not None:
            if hasattr(function, "_semantikon_metadata"):
                metadata.update(function._semantikon_metadata)
            function_data = get_function_dict(function)
            function_data["identifier"] = ".".join(
                (
                    function_data["module"],
                    function_data["qualname"],
                    function_data["version"],
                )
            )
            metadata["function"] = function_data
        G.add_node(node_name, **metadata)

        output_labels = list(node_data.output_ports)
        if len(output_labels) == 1 and output_labels[0] == "output_0":
            output_labels = ["output"]

        for position, (label, port) in enumerate(node_data.input_ports.items()):
            inp_name = Input(node=node_name, port=label)
            io_data: dict[str, Any] = {"position": position}
            if not isinstance(port.value, fr.schemas.NotData):
                io_data["value"] = port.value
            if type_hint := annotation_to_type_hint(port.annotation):
                io_data["dtype"] = type_hint
            if type_metadata := annotation_to_type_metadata(port.annotation):
                io_data.update(type_metadata.to_dictionary())
            if not isinstance(port.default, fr.schemas.NotData):
                io_data["default"] = port.default
            G.add_node(inp_name, **io_data)
            G.add_edge(inp_name, node_name)
        for position, (raw_label, port) in enumerate(node_data.output_ports.items()):
            label = output_labels[position] if raw_label == "output_0" else raw_label
            out_name = Output(node=node_name, port=label)
            io_data = {"position": position}
            if not isinstance(port.value, fr.schemas.NotData):
                io_data["value"] = port.value
            if type_hint := annotation_to_type_hint(port.annotation):
                io_data["dtype"] = type_hint
            if type_metadata := annotation_to_type_metadata(port.annotation):
                io_data.update(type_metadata.to_dictionary())
            G.add_node(out_name, **io_data)
            G.add_edge(node_name, out_name)

        if not isinstance(node_data, fr.schemas.DagData):
            return

        recipe = node_data.recipe
        for child_label, child in node_data.nodes.items():
            child_name = Node(name=child_label, parent=node_name)
            _add_node(
                child,
                child_name,
                parent_name=node_name,
                workflow_label=(
                    Node(child_label) if isinstance(child, fr.schemas.DagData) else None
                ),
            )

        child_recipes = recipe.nodes
        for target, source in recipe.input_edges.items():
            G.add_edge(
                Input(node=node_name, port=source.port),
                Input(node=Node(parent=node_name, name=target.node), port=target.port),
            )
        for target, source in recipe.edges.items():
            child_outputs = list(child_recipes[source.node].outputs)
            src_port = _output_port_label(source.port, child_outputs)
            G.add_edge(
                Output(node=Node(parent=node_name, name=source.node), port=src_port),
                Input(node=Node(parent=node_name, name=target.node), port=target.port),
            )
        for target, source in recipe.output_edges.items():
            target_port = _output_port_label(target.port, list(recipe.outputs))
            if isinstance(source, fr.schemas.InputSource):
                G.add_edge(
                    Input(node=node_name, port=source.port),
                    Output(node=node_name, port=target_port),
                )
            else:
                child_outputs = list(child_recipes[source.node].outputs)
                src_port = _output_port_label(source.port, child_outputs)
                G.add_edge(
                    Output(node=Node(parent=node_name, name=source.node), port=src_port),
                    Output(node=node_name, port=target_port),
                )

    _add_node(workflow, Node(root_label), workflow_label=Node(root_label))
    return G


def _get_hashed_node_dict_from_graph(G: SemantikonDiGraph) -> dict[str, dict[str, Any]]:
    hash_dict: dict[str, dict[str, Any]] = {}
    for node in nx.topological_sort(G):
        data = G.nodes[node]
        if isinstance(node, IO):
            for term in ("hash", "value"):
                if term in data:
                    continue
                for predecessor in G.predecessors(node):
                    predecessor_data = G.nodes[predecessor]
                    if term in predecessor_data:
                        data[term] = predecessor_data[term]
                        break
            continue

        hash_dict_tmp: dict[str, Any] = {
            "inputs": {},
            "outputs": [
                G.nodes[out].get("label", out.port) for out in G.successors(node)
            ],
            "node": copy.deepcopy(data.get("function")),
        }
        if hash_dict_tmp["node"] is None:
            continue
        hash_dict_tmp["node"]["connected_inputs"] = []
        missing_input = False
        for inp in G.predecessors(node):
            inp_data = G.nodes[inp]
            if "hash" in inp_data:
                hash_dict_tmp["inputs"][inp.port] = inp_data["hash"]
                hash_dict_tmp["node"]["connected_inputs"].append(inp.port)
            elif "value" in inp_data or "default" in inp_data:
                value = inp_data.get("value", inp_data.get("default"))
                if is_dataclass(value) and not isinstance(value, type):
                    hash_dict_tmp["inputs"][inp.port] = asdict(value)
                else:
                    hash_dict_tmp["inputs"][inp.port] = value
            else:
                missing_input = True
                break
        if missing_input:
            continue
        h = sha256(
            json.dumps(hash_dict_tmp, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for out in G.successors(node):
            G.nodes[out]["hash"] = h + "@" + G.nodes[out].get("label", out.port)
        hash_dict_tmp["hash"] = h
        hash_dict[node] = hash_dict_tmp
    return hash_dict


def _remove_constant(G: SemantikonDiGraph) -> None:
    to_delete = []
    for node, data in G.nodes.data():
        if isinstance(node, Node) and data["type"] == "constant":
            output_node = next(iter(G.successors(node)))
            const_value = G.nodes[output_node]["value"]
            to_delete.extend([node, output_node])
            for inp in G.successors(output_node):
                G.nodes[inp]["constant_value"] = const_value
    G.remove_nodes_from(to_delete)


def serialize_and_convert_to_networkx(
    workflow: dict | fr.schemas.DagData | fr.schemas.WorkflowRecipe,
    hash_data: bool = True,
    prefix: str | None = None,
) -> SemantikonDiGraph:
    """
    Serialize a flowrep workflow into a SemantikonDiGraph, optionally
    hashing node data.

    Args:
        workflow (dict | DagData | WorkflowRecipe): Workflow representation.
        hash_data (bool): Whether to hash node data.
        prefix (str | None): Optional fixed prefix for type-level namespace
            fragments.

    Returns:
        SemantikonDiGraph: The serialized workflow graph.
    """
    if isinstance(workflow, dict):
        warnings.warn(
            "Passing a dict to 'serialize_and_convert_to_networkx' is deprecated"
            " and will be removed in a future version. Please pass a 'flowrep'"
            " object instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        workflow = dict_to_nodedata(workflow)
    if isinstance(workflow, fr.schemas.WorkflowRecipe):
        workflow = fr.schemas.DagData.from_recipe(workflow)
    if not isinstance(workflow, fr.schemas.DagData):
        raise TypeError(
            f"Invalid workflow type. Expected dict, flowrep"
            f" {fr.schemas.DagData.__name__!r}, or flowrep"
            f" {fr.schemas.WorkflowRecipe.__name__!r}, but got {type(workflow)}."
        )

    G = _workflow_to_networkx(workflow, prefix=prefix)
    if hash_data:
        try:
            hashed_dict = _get_hashed_node_dict_from_graph(G)
        except Exception as e:
            raise RuntimeError(
                "Failed to hash workflow data - use only hashable inputs or set"
                " hash_data=False"
            ) from e
        for node, data in hashed_dict.items():
            G.append_hash(node, data["hash"])
    _remove_constant(G)
    return G


class _HashGraph:
    def _normalize(self, obj: Any) -> Any:
        """
        Convert objects into a deterministic, JSON-safe representation.
        """
        if is_dataclass(obj) and not isinstance(obj, type):
            return {k: self._normalize(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: self._normalize(v) for k, v in obj.items()}
        elif isinstance(obj, IdentifiedNode):
            if isinstance(obj, BNode):
                raise TypeError("Blank nodes cannot be normalized for hashing.")
            return {"__type__": "URIRef", "value": self._normalize(str(obj))}
        elif isinstance(obj, (list, tuple)):
            return [self._normalize(v) for v in obj]
        elif isinstance(obj, str):
            return unicodedata.normalize("NFC", obj)
        elif isinstance(obj, (int, float, bool)) or obj is None:
            return obj
        else:
            raise TypeError(
                f"Unsupported type for normalization: {type(obj)} of value {obj}"
            )

    def _canonical_json(self, data: dict) -> str:
        """
        Deterministic JSON serialization.
        """
        return json.dumps(
            self._normalize(data),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def _get_graph_hash(
        self, G: SemantikonDiGraph, with_global_inputs: bool = True
    ) -> str:
        """
        Generate a deterministic hash for a graph, independent of OS,
        Python version, and non-serializable runtime values.
        """
        G_tmp = nx.DiGraph()

        for node in G.nodes:
            attrs = {}
            if "function" in G.nodes[node]:
                func = G.nodes[node]["function"]
                attrs["function"] = func.get("hash") or func.get("identifier")
            if G.in_degree(node) == 0 and with_global_inputs:
                if "value" in G.nodes[node]:
                    attrs["value"] = G.nodes[node]["value"]
                elif "default" in G.nodes[node]:
                    attrs["value"] = G.nodes[node]["default"]
            if isinstance(node, IO):
                attrs["port"] = node.port
            G_tmp.add_node(node, canon=self._canonical_json(attrs))
        for u, v in G.edges:
            G_tmp.add_edge(u, v)

        return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
            G_tmp,
            node_attr="canon",
        )


def _get_graph_hash(G: SemantikonDiGraph, with_global_inputs: bool = True) -> str:
    """
    Generate a hash for a NetworkX graph, making sure that data types and
    values (except for the global ones) because they can often not be
    serialized.

    Args:
        G (SemantikonDiGraph): input graph
        with_global_inputs (bool): if True, keep values for global inputs

    Returns:
        (str): hash of the graph
    """
    hasher = _HashGraph()
    return hasher._get_graph_hash(G, with_global_inputs)
