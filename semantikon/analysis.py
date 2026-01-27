from __future__ import annotations
import string
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Iterable

import networkx as nx
from rdflib import OWL, RDFS, Graph, Literal, URIRef

from semantikon.converter import to_identifier
from semantikon.ontology import BASE, SNS, serialize_and_convert_to_networkx


def label_to_uri(graph: Graph, label: str) -> list[URIRef]:
    """
    Convert a human-readable label to its corresponding URIRef in the graph.

    Args:
        graph (Graph): The RDF graph to query.
        label (str): The human-readable label or URIRef.

    Returns:
        list[URIRef]: The corresponding URIs from the graph.
    """
    query = """SELECT ?s
    WHERE {
      ?s rdfs:label ?label .
      ?s a owl:Class .
    }"""
    result = list(graph.query(query, initBindings={"label": Literal(label)}))
    assert len(result) > 0, f"No result found for {label}"
    return [r[0] for r in result]


def request_values(wf_dict: dict, graph: Graph) -> dict:
    """
    Given a workflow dictionary and an RDF graph, this function
    populates the workflow dictionary with values extracted from the graph
    based on hash identifiers.

    Args:
        wf_dict (dict): The workflow dictionary to populate.
        graph (Graph): The RDF graph containing data nodes.

    Returns:
        dict: The updated workflow dictionary with populated values.
    """
    G = serialize_and_convert_to_networkx(wf_dict)

    # Collect all hashes that need values, along with their target locations.
    hash_nodes: list[dict[str, Any]] = []
    hashes: set[str] = set()

    for node, data in G.nodes.data():
        if data.get("step") == "node":
            continue
        if "hash" in data and "value" not in data:
            node_hash = data["hash"]
            hashes.add(node_hash)
            keys = node.split("-")[1:]
            hash_nodes.append(
                {
                    "hash": node_hash,
                    "keys": keys,
                }
            )

    # If there are no hashes to resolve, return early.
    if not hashes:
        return wf_dict

    # Build a single SPARQL query that retrieves values for all hashes at once.
    values_str = " ".join(f'"{h}"' for h in hashes)
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX iao: <http://purl.obolibrary.org/obo/IAO_>
    SELECT ?h ?v WHERE {{
        ?h_bnode rdf:value ?h .
        ?data_node iao:0000235 ?h_bnode .
        ?data_node rdf:value ?v .
        VALUES ?h {{ {values_str} }}
    }}
    """

    # Execute the batched query and build a mapping from hash to value.
    hash_to_value: dict[str, Any] = {}
    for row in graph.query(query):
        h_val = row[0].toPython()
        v_val = row[1].toPython()
        # Preserve existing behavior: only the first value per hash is used.
        if h_val not in hash_to_value:
            hash_to_value[h_val] = v_val

    # Populate wf_dict with the retrieved values.
    for item in hash_nodes:
        h = item["hash"]
        keys = item["keys"]
        if h not in hash_to_value:
            continue
        value = hash_to_value[h]
        if len(keys) == 3:
            wf_dict["nodes"][keys[0]][keys[1]][keys[2]]["value"] = value
        elif len(keys) == 2:
            wf_dict[keys[0]][keys[1]]["value"] = value
    return wf_dict


class TrieNode:
    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.terminal = False


class _Node:
    __slots__ = ("_node", "_path", "_graph")

    def __init__(self, node: TrieNode, path: Iterable[str], graph: Graph):
        self._node = node
        self._path = tuple(path)
        self._graph = graph

    def __getattr__(self, name: str):
        if name not in self._node.children:
            raise AttributeError(name)
        return _Node(self._node.children[name], self._path + (name,), self._graph)

    def __dir__(self):
        if self._node.terminal:
            return ["query", "to_query_text"]
        return sorted(self._node.children.keys())

    def query(self, fallback_to_hash: bool = False) -> list[tuple[Any, ...]]:
        """
        Execute a SPARQL query for this node against the bound graph.

        Args:
            fallback_to_hash (bool): If True, missing values would be filled
                with their hash values, potentially giving the user to look
                up the data elsewhere.

        Returns:
            list[tuple[Any, ...]]: The query results as a list of tuples.
        """
        qn = _QueryHolder([self], self._graph)
        return qn.query(fallback_to_hash=fallback_to_hash)

    def to_query_text(self) -> str:
        """Generate a SPARQL query string for this node."""
        qn = _QueryHolder([self], self._graph)
        return qn.to_query_text()

    def value(self) -> URIRef:
        return BASE["-".join(self._path)]

    def __and__(self, other: _Node | URIRef | _QueryHolder) -> _QueryHolder:
        if isinstance(other, _Node) or isinstance(other, URIRef):
            nodes = [self, other]
        else:
            assert isinstance(other, _QueryHolder)
            nodes = [self] + other._nodes
        return _QueryHolder(nodes, self._graph)

    def __rand__(self, other: URIRef) -> _QueryHolder:
        assert isinstance(other, URIRef), type(other)
        return _QueryHolder([other, self], self._graph)


@dataclass
class _QueryHolder:
    """Container for one or more query nodes bound to an RDF graph.

    This helper encapsulates a collection of `_Node` instances together with
    the RDFLib :class:`Graph` they belong to and provides a small API for
    building and executing SPARQL queries.

    Attributes
    ----------
    _nodes:
        List of `_Node` instances that define the pattern of the query to be
        generated.
    _graph:
        The RDFLib :class:`Graph` against which the generated query will be
        constructed and executed.
    """

    _nodes: list[_Node]
    _graph: Graph

    def to_query_graph(self, fallback_to_hash: bool = False) -> Graph:
        """Build the intermediate query graph for the held nodes.

        Args:
            fallback_to_hash (bool): If True, missing values would be filled
                with their hash values, potentially giving the user to look
                up the data elsewhere.

        Returns:
            Graph: An RDFLib :class:`Graph` representing the query structure for
                the held nodes.
        """
        sw = SparqlWriter(self._graph)
        return sw.get_query_graph(*self._nodes, fallback_to_hash=fallback_to_hash)

    def to_query_text(self, fallback_to_hash: bool = False) -> str:
        """Generate a SPARQL query string for the held nodes.

        The method first builds the intermediate query graph via
        :meth:`to_query_graph` and then converts it to a textual SPARQL
        representation.

        Args:
            fallback_to_hash (bool): If True, missing values would be filled
                with their hash values, potentially giving the user to look
                up the data elsewhere.

        Returns:
            str: The generated SPARQL query string.
        """
        G = self.to_query_graph(fallback_to_hash=fallback_to_hash)
        return SparqlWriter.get_query_text(G)

    def query(self, fallback_to_hash: bool = False) -> list[tuple[Any, ...]]:
        """Execute the generated SPARQL query against the stored graph.

        The query text produced by :meth:`to_query_text` is run with
        :meth:`Graph.query`, and each bound value in the result set is
        converted to a native Python object via its ``toPython`` method.

        Args:
            fallback_to_hash (bool): If True, missing values would be filled
                with their hash values, potentially giving the user to look
                up the data elsewhere.

        Returns:
            list[tuple[Any, ...]]: The query results as a list of tuples.
        """
        text = self.to_query_text(fallback_to_hash=fallback_to_hash)
        return [
            tuple([a if a is None else a.toPython() for a in item])
            for item in self._graph.query(text)
        ]

    def __and__(self, other: _Node | URIRef | _QueryHolder) -> _QueryHolder:
        if isinstance(other, _Node) or isinstance(other, URIRef):
            nodes = self._nodes + [other]
        else:
            assert isinstance(other, _QueryHolder)
            nodes = self._nodes + other._nodes
        return _QueryHolder(nodes, self._graph)

    def __rand__(self, other: URIRef) -> _QueryHolder:
        assert isinstance(other, URIRef)
        return _QueryHolder([other] + self._nodes, self._graph)


class Completer(_Node):
    def __init__(self, values: Iterable[str], graph: Graph):
        root = TrieNode()
        for value in values:
            node = root
            for part in value.split("-"):
                node = node.children.setdefault(part, TrieNode())
            node.terminal = True

        super().__init__(root, (), graph)


def query_io_completer(graph: Graph) -> Completer:
    """
    Create a Completer for input/output ports in the given graph.

    Args:
        graph (Graph): An RDFLib graph containing the ontology or data to query.

    Returns:
        Completer: A Completer instance for input/output ports.
    """
    all_ios = []
    for pred in ["pmd:0000066", "pmd:0000067"]:
        query = f"""
        PREFIX pmd: <https://w3id.org/pmd/co/PMD_>
        SELECT ?io WHERE {{
            ?io rdfs:subClassOf {pred} .
        }}"""
        for g in graph.query(query):
            all_ios.append(g[0].split("/")[-1])
    return Completer(all_ios, graph)


class SparqlWriter:
    """
    A class for generating and executing SPARQL queries based on a graph structure.
    """

    def __init__(self, graph: Graph):
        """
        Initialize the SparqlWriter with a given RDFLib graph.

        Args:
            graph (Graph): An RDFLib graph containing the ontology or data to query.
        """
        self._graph = graph

    @cached_property
    def G(self) -> nx.DiGraph:
        """
        Construct a directed graph (DiGraph) representation of the ontology.

        The graph is built by querying the RDFLib graph for subclass relationships
        and OWL restrictions. Each edge in the graph represents a relationship
        between a parent and child class, with the predicate stored as edge data.

        Returns:
            nx.DiGraph: A directed graph representing the ontology structure.
        """
        query = """
        SELECT ?parent ?property ?child WHERE {
            ?parent rdfs:subClassOf ?bnode .
            ?bnode a owl:Restriction .
            ?bnode owl:onProperty ?property .
            ?bnode owl:someValuesFrom ?child .
        }"""
        G = nx.DiGraph()
        for subj, pred, obj in self._graph.query(query):
            G.add_edge(subj, obj, predicate=pred)
        return G

    def _is_io_port(self, node: URIRef) -> bool:
        return any(
            (node, RDFS.subClassOf, p) in self._graph
            for p in [SNS.input_assignment, SNS.output_assignment]
        )

    def _to_qname(self, term: URIRef) -> str:
        return self._graph.qname(term)

    def _get_head_node(self, data_nodes):
        candidates = list(self._graph.subjects(RDFS.subClassOf, SNS.workflow_node))
        for node in nx.topological_sort(self.G):
            if node in candidates and all(
                nx.has_path(self.G, node, dn) for dn in data_nodes
            ):
                return node
        raise ValueError("No common head node found")

    def get_query_graph(self, *args, fallback_to_hash: bool = False) -> nx.DiGraph:
        """
        Generate a query graph based on the provided arguments.

        The query graph is a directed graph (DiGraph) where nodes represent
        data elements and edges represent relationships between them. This graph
        can be used to generate SPARQL query text.

        Args:
            *args: A variable number of arguments representing nodes in the graph.
                   Each argument can be an RDFLib node or a value.
            fallback_to_hash (bool): If True, missing values would be filled
                with their hash values, potentially giving the user to look
                up the data elsewhere.

        Returns:
            nx.DiGraph: A directed graph representing the query structure.
        """
        G = nx.DiGraph()
        data_nodes = []
        for ii, arg in enumerate(args):
            if isinstance(arg, _Node):
                arg = arg.value()
            while self._is_io_port(arg):
                arg = list(self.G.successors(arg))[0]
            data_nodes.append(arg)
            value_node = self._to_qname(data_nodes[-1] + "_value")
            hash_node = self._to_qname(data_nodes[-1] + "_hash")
            if fallback_to_hash:
                G.add_node(value_node, output=string.ascii_lowercase[ii] + "_a")
                G.add_node(hash_node, output=string.ascii_lowercase[ii] + "_b")
            else:
                G.add_node(value_node, output=string.ascii_lowercase[ii])
            G.add_node(data_nodes[-1])
            G.add_edge(self._to_qname(data_nodes[-1]), data_nodes[-1], predicate="a")
            if fallback_to_hash:
                G.add_edge(
                    self._to_qname(data_nodes[-1]),
                    value_node,
                    predicate="rdf:value",
                    optional=value_node,
                )
                G.add_edge(
                    self._to_qname(data_nodes[-1]),
                    hash_node + "_b",
                    predicate=f"<{SNS.denoted_by}>",
                    optional=hash_node,
                )
                G.add_edge(
                    hash_node + "_b",
                    hash_node,
                    predicate="rdf:value",
                    optional=hash_node,
                )
            else:
                G.add_edge(
                    self._to_qname(data_nodes[-1]),
                    value_node,
                    predicate="rdf:value",
                )
        if len(data_nodes) > 1:
            head_node = self._get_head_node(data_nodes)
            for node in data_nodes:
                path = nx.shortest_path(self.G, head_node, node)
                assert len(path) > 1
                for u, v in zip(path[:-1], path[1:]):
                    if not self.G.has_edge(u, v):
                        u, v = v, u
                    G.add_edge(
                        self._to_qname(u),
                        self._to_qname(v),
                        predicate=self.G.edges[u, v]["predicate"],
                    )
        return G

    @staticmethod
    def get_query_text(G: nx.DiGraph) -> str:
        """
        Convert a query graph into SPARQL query text.

        This method takes a directed graph (DiGraph) representing a query structure
        and generates the corresponding SPARQL query text.

        Args:
            G (nx.DiGraph): A directed graph representing the query structure.

        Returns:
            str: The SPARQL query text.
        """

        def _to_sparql_line(subj, pred, obj):
            subj, obj = [
                f"<{e}>" if isinstance(e, URIRef) else f"?{to_identifier(e)}"
                for e in [subj, obj]
            ]
            pred = (
                f"<{data['predicate']}>"
                if isinstance(data["predicate"], URIRef)
                else data["predicate"]
            )
            return f"{subj} {pred} {obj} ."

        output_with_ind = defaultdict(list)
        for key, value in dict(
            sorted(
                {
                    data["output"]: f"?{to_identifier(node)}"
                    for node, data in G.nodes.data()
                    if "output" in data
                }.items()
            )
        ).items():
            output_with_ind[key.split("_")[0]].append(value)
        select_line = "SELECT"
        for key, value in output_with_ind.items():
            if len(value) == 1:
                select_line += " " + value[0]
            else:
                select_line += " (COALESCE(" + ", ".join(value) + f") AS ?{key})"
        select_line += " WHERE {"
        optional_dict = defaultdict(list)
        lines = [select_line]
        for subj, obj, data in G.edges.data():
            if "optional" in data:
                optional_dict[data["optional"]].append(
                    _to_sparql_line(subj, data["predicate"], obj)
                )
            else:
                lines.append("  " + _to_sparql_line(subj, data["predicate"], obj))
        if len(optional_dict) > 0:
            for o_line in optional_dict.values():
                lines.append("  OPTIONAL { " + "\n".join(o_line) + " }")
        lines.append("}")
        return "\n".join(lines)
