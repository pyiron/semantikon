from __future__ import annotations

import copy
import json
import unicodedata
from dataclasses import asdict, is_dataclass
from functools import cache, cached_property
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


class SemantikonDiGraph(nx.DiGraph):
    """Workflow graph with deterministic namespace fragments.

    The graph only stores suffix fragments (e.g. ``abc123_``) for type- and
    assertion-level identifiers. Ontology-specific base namespaces are applied
    later by the ontology serialization layer.
    """

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

    @cache
    def _get_data_node(self, io: str) -> str:
        while True:
            candidate = [
                c for c in self.predecessors(io) if self.nodes[c]["step"] != "node"
            ]
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
                (e.g., "label" or "arg"). Defaults to None.

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
                if self.nodes[child]["step"] == "node":
                    continue

                child_label = current_label
                if child_label is None:
                    child_label = self.nodes[child].get(
                        "label", self.nodes[child]["arg"]
                    )

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


def _port_to_dict(
    *,
    value: Any,
    annotation: Any,
    default: Any = fr.schemas.NOT_DATA,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not isinstance(value, fr.schemas.NotData):
        data["value"] = value
    if type_hint := annotation_to_type_hint(annotation):
        data["dtype"] = type_hint
    if type_metadata := annotation_to_type_metadata(annotation):
        data.update(type_metadata.to_dictionary())
    if not isinstance(default, fr.schemas.NotData):
        data["default"] = default
    return data


def _node_data_to_metadata(
    data: fr.schemas.NodeData,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    function = None
    if isinstance(data, fr.schemas.AtomicData):
        metadata["type"] = "atomic"
        function = data.function
    elif isinstance(data, fr.schemas.DagData):
        metadata["type"] = "workflow"
        if label is not None:
            metadata["label"] = label
        if data.recipe.reference is not None:
            function = retrieve.import_from_string(
                data.recipe.reference.info.fully_qualified_name
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
    return metadata


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
        node_name: str,
        *,
        parent_name: str | None = None,
        workflow_label: str | None = None,
    ):
        node_attrs = _node_data_to_metadata(node_data, label=workflow_label)
        if parent_name is not None:
            node_attrs["parent"] = parent_name
        G.add_node(node_name, step="node", **node_attrs)

        output_labels = list(node_data.output_ports)
        if len(output_labels) == 1 and output_labels[0] == "output_0":
            output_labels = ["output"]

        for position, (label, port) in enumerate(node_data.input_ports.items()):
            io_name = f"{node_name}-inputs-{label}"
            io_data = _port_to_dict(
                value=port.value,
                annotation=port.annotation,
                default=port.default,
            )
            G.add_node(io_name, step="inputs", arg=label, position=position, **io_data)
            G.add_edge(io_name, node_name)
        for position, (raw_label, port) in enumerate(node_data.output_ports.items()):
            label = output_labels[position] if raw_label == "output_0" else raw_label
            io_name = f"{node_name}-outputs-{label}"
            io_data = _port_to_dict(value=port.value, annotation=port.annotation)
            G.add_node(io_name, step="outputs", arg=label, position=position, **io_data)
            G.add_edge(node_name, io_name)

        if not isinstance(node_data, fr.schemas.DagData):
            return

        recipe = node_data.recipe
        for child_label, child in node_data.nodes.items():
            child_name = f"{node_name}-{child_label}"
            _add_node(
                child,
                child_name,
                parent_name=node_name,
                workflow_label=(
                    child_label if isinstance(child, fr.schemas.DagData) else None
                ),
            )

        child_recipes = recipe.nodes
        for target, source in recipe.input_edges.items():
            G.add_edge(
                f"{node_name}-inputs-{source.port}",
                f"{node_name}-{target.node}-inputs-{target.port}",
            )
        for target, source in recipe.edges.items():
            child_outputs = list(child_recipes[source.node].outputs)
            src_port = _output_port_label(source.port, child_outputs)
            G.add_edge(
                f"{node_name}-{source.node}-outputs-{src_port}",
                f"{node_name}-{target.node}-inputs-{target.port}",
            )
        for target, source in recipe.output_edges.items():
            target_port = _output_port_label(target.port, list(recipe.outputs))
            if isinstance(source, fr.schemas.InputSource):
                G.add_edge(
                    f"{node_name}-inputs-{source.port}",
                    f"{node_name}-outputs-{target_port}",
                )
            else:
                child_outputs = list(child_recipes[source.node].outputs)
                src_port = _output_port_label(source.port, child_outputs)
                G.add_edge(
                    f"{node_name}-{source.node}-outputs-{src_port}",
                    f"{node_name}-outputs-{target_port}",
                )

    _add_node(workflow, root_label, workflow_label=root_label)
    return G


def _get_hashed_node_dict_from_graph(G: SemantikonDiGraph) -> dict[str, dict[str, Any]]:
    hash_dict: dict[str, dict[str, Any]] = {}
    for node in nx.topological_sort(G):
        data = G.nodes[node]
        if data.get("step") != "node":
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
                G.nodes[out].get("label", out.split("-")[-1])
                for out in G.successors(node)
            ],
            "node": copy.deepcopy(data.get("function")),
        }
        if hash_dict_tmp["node"] is None:
            continue
        hash_dict_tmp["node"]["connected_inputs"] = []
        missing_input = False
        for inp in G.predecessors(node):
            inp_data = G.nodes[inp]
            inp_name = inp.split("-")[-1]
            if "hash" in inp_data:
                hash_dict_tmp["inputs"][inp_name] = inp_data["hash"]
                hash_dict_tmp["node"]["connected_inputs"].append(inp_name)
            elif "value" in inp_data:
                value = inp_data["value"]
                if is_dataclass(value) and not isinstance(value, type):
                    hash_dict_tmp["inputs"][inp_name] = asdict(value)
                else:
                    hash_dict_tmp["inputs"][inp_name] = value
            else:
                missing_input = True
                break
        if missing_input:
            continue
        h = sha256(
            json.dumps(hash_dict_tmp, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for out in G.successors(node):
            G.nodes[out]["hash"] = (
                h + "@" + G.nodes[out].get("label", out.split("-")[-1])
            )
        hash_dict_tmp["hash"] = h
        hash_dict[node] = hash_dict_tmp
    return hash_dict


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
        workflow = dict_to_nodedata(workflow)
    if isinstance(workflow, fr.schemas.WorkflowRecipe):
        workflow = fr.schemas.DagData.from_recipe(workflow)
    if not isinstance(workflow, fr.schemas.DagData):
        raise TypeError(
            f"Invalid workflow type. Expected dict, flowrep {fr.schemas.DagData.__name__!r}, or "
            f"flowrep {fr.schemas.WorkflowRecipe.__name__!r}, but got {type(workflow)}."
        )

    G = _workflow_to_networkx(workflow, prefix=prefix)
    if hash_data:
        try:
            hashed_dict = _get_hashed_node_dict_from_graph(G)
        except Exception as e:
            raise RuntimeError(
                "Failed to hash workflow data - use only hashable inputs or set hash_data=False"
            ) from e
        for node, data in hashed_dict.items():
            G.append_hash(node, data["hash"])
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
                attrs["hash"] = G.nodes[node]["function"].get(
                    "hash", G.nodes[node]["function"].get("identifier")
                )
            if G.in_degree(node) == 0 and with_global_inputs:
                if "value" in G.nodes[node]:
                    attrs["value"] = G.nodes[node]["value"]
                elif "default" in G.nodes[node]:
                    attrs["value"] = G.nodes[node]["default"]
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
