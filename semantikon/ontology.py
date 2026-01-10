from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, is_dataclass
from functools import cache, cached_property
from hashlib import sha256
from typing import Any, Callable, Dict, Iterable, TypeAlias, cast

import networkx as nx
from flowrep.workflow import get_hashed_node_dict
from owlrl import DeductiveClosure, RDFS_Semantics
from pyshacl import validate
from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import SH
from rdflib.term import IdentifiedNode

from semantikon.converter import (
    get_function_dict,
    meta_to_dict,
    parse_input_args,
    parse_output_args,
    to_identifier,
)
from semantikon.metadata import SemantikonURI
from semantikon.qudt import UnitsDict

IAO: Namespace = Namespace("http://purl.obolibrary.org/obo/IAO_")
QUDT: Namespace = Namespace("http://qudt.org/schema/qudt/")
RO: Namespace = Namespace("http://purl.obolibrary.org/obo/RO_")
BFO: Namespace = Namespace("http://purl.obolibrary.org/obo/BFO_")
OBI: Namespace = Namespace("http://purl.obolibrary.org/obo/OBI_")
PMD: Namespace = Namespace("https://w3id.org/pmd/co/PMD_")
SCHEMA: Namespace = Namespace("http://schema.org/")
STATO: Namespace = Namespace("http://purl.obolibrary.org/obo/STATO_")
BASE: Namespace = Namespace("http://pyiron.org/ontology/")


@dataclass(frozen=True)
class SNS:
    has_part: URIRef = BFO["0000051"]
    part_of: URIRef = BFO["0000050"]
    participates_in: URIRef = RO["0000056"]
    has_participant: URIRef = RO["0000057"]
    has_specified_input: URIRef = OBI["0000293"]
    has_specified_output: URIRef = OBI["0000299"]
    input_assignment: URIRef = PMD["0000066"]
    executes: URIRef = STATO["0000102"]
    output_assignment: URIRef = PMD["0000067"]
    precedes: URIRef = BFO["0000063"]
    process: URIRef = BFO["0000015"]
    continuant: URIRef = BFO["0000002"]
    value_specification: URIRef = OBI["0001933"]
    specifies_value_of: URIRef = OBI["0001927"]
    derives_from: URIRef = RO["0001000"]
    software_method: URIRef = IAO["0000591"]
    textual_entity: URIRef = IAO["0000300"]
    denoted_by: URIRef = IAO["0000235"]
    identifier: URIRef = IAO["0020000"]
    is_about: URIRef = IAO["0000136"]
    input_specification: URIRef = BASE["input_specification"]
    output_specification: URIRef = BASE["output_specification"]
    has_parameter_specification: URIRef = BASE["has_parameter_specification"]
    has_parameter_position: URIRef = BASE["has_parameter_position"]
    has_default_literal_value: URIRef = BASE["has_default_literal_value"]
    has_constraint: URIRef = BASE["has_constraint"]


ud = UnitsDict()

_triple_type: TypeAlias = list[
    tuple[IdentifiedNode | str | None, URIRef, IdentifiedNode | str | None]
]


_rest_type: TypeAlias = tuple[tuple[URIRef, URIRef], ...]


def _units_to_uri(units: str | URIRef) -> URIRef:
    if isinstance(units, URIRef):
        return units
    key = ud[units]
    if key is not None:
        return key
    return URIRef(units)


class SemantikonDiGraph(nx.DiGraph):
    @cached_property
    def t_ns(self):
        h = (
            "W" + _get_graph_hash(self, with_global_inputs=False)[:8]
            if self.graph["prefix"] is None
            else self.graph["prefix"]
        )
        return Namespace(BASE + h + "_")

    @cached_property
    def a_ns(self):
        h = _get_graph_hash(self, with_global_inputs=True)
        return Namespace(BASE + h + "_")

    def get_a_node(self, node_name: str) -> BNode:
        return BNode(self.a_ns[node_name])

    @cache
    def _get_data_node(self, io: str) -> str:
        while True:
            candidate = list(self.predecessors(io))
            assert len(candidate) <= 1
            if len(candidate) == 0 or self.nodes[candidate[0]]["step"] == "node":
                return f"{io}_data"
            io = candidate[0]

    def append_hash(
        self,
        node: str,
        hash_value: str,
        label: str | None = None,
        remove_data: bool = False,
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
            remove_data (bool, optional): If True, removes the "value" field
                from the nodes' data during the traversal. Defaults to False.

        Notes:
            - The function uses an iterative approach to avoid recursion,
                making it suitable for graphs with deep hierarchies.
            - The hash value for each node is updated in the format:
                `parent_hash@child_label`.

        """
        # Use a stack to keep track of nodes to process
        stack = [(node, hash_value, label)]

        while stack:
            current_node, current_hash, current_label = stack.pop()

            for child in self.successors(current_node):
                if self.nodes[child]["step"] == "node":
                    continue

                # Determine the label for this specific child
                child_label = current_label
                if child_label is None:
                    child_label = self.nodes[child].get(
                        "label", self.nodes[child]["arg"]
                    )

                # Update the hash for the child node
                self.nodes[child]["hash"] = current_hash + f"@{child_label}"

                # Optionally remove the "value" data
                if remove_data and "value" in self.nodes[child]:
                    del self.nodes[child]["value"]

                # Add the child to the stack for further processing
                stack.append((child, current_hash, child_label))


def _inherit_properties(graph: Graph, n_max: int = 1000):
    query = f"""\
    PREFIX rdfs: <{RDFS}>
    PREFIX rdf: <{RDF}>
    PREFIX owl: <{OWL}>
    PREFIX ro: <{RO}>
    INSERT {{
        ?subject ?p ?o .
    }}
    WHERE {{
        ?subject ro:0001000 ?target .
        ?target ?p ?o .
        FILTER(?p != ro:0001000)
        FILTER(?p != rdfs:label)
        FILTER(?p != rdf:value)
        FILTER(?p != rdf:type)
        FILTER(?p != owl:sameAs)
    }}
    """
    n = 0
    for _ in range(n_max):
        graph.update(query)
        if len(graph) == n:
            break
        n = len(graph)


def _extract_shacl_shapes(input_graph: Graph) -> Graph:
    """
    Extract all SHACL NodeShapes and the full subgraph of constraints
    reachable from them.
    """
    output_graph = _get_bound_graph()
    visited = set()

    def copy_subgraph(node):
        if node in visited:
            return
        visited.add(node)

        for p, o in input_graph.predicate_objects(node):
            output_graph.add((node, p, o))
            # Recurse into blank nodes and referenced shapes
            if isinstance(o, BNode) or (isinstance(o, URIRef) and o.startswith(SH)):
                copy_subgraph(o)

    for shape in input_graph.subjects(RDF.type, SH.NodeShape):
        copy_subgraph(shape)

    return output_graph


def validate_values(
    graph: Graph,
    run_reasoner: bool = True,
    copy_graph: bool = True,
    strict_typing: bool = True,
) -> tuple:
    """
    Validate if all values required by restrictions are present in the graph

    Args:
        graph (Graph): input RDF graph
        run_reasoner (bool): if True, run OWL RL reasoner before validation
        copy_graph (bool): if True, work on a copy of the graph. When the graph
            is not copied and reasoner is run, the input graph is modified by
            inferred triples.
        strict_typing (bool): if True, enforce strict typing, i.e. if the
            URI/units of the input of a node are defined, the output of the
            preceding node must have the same URI/units

    Returns:
        (tuple): validation result and message from pyshacl
    """
    g = copy.deepcopy(graph) if copy_graph and run_reasoner else graph
    excluded = []
    if not strict_typing:
        excluded = _get_undefined_connections(g, "qudt:hasUnit")
        excluded.extend(_get_undefined_connections(g, "obi:0001927"))
    shacl = owl_restrictions_to_shacl(g, excluded_nodes=excluded)
    shacl += _extract_shacl_shapes(g)
    if run_reasoner:
        DeductiveClosure(RDFS_Semantics).expand(g)
        _inherit_properties(g)
    return validate(g, shacl_graph=shacl)


def _get_bound_graph(*args, **kwargs):
    graph = Graph(*args, **kwargs)
    graph.bind("qudt", str(QUDT))
    graph.bind("unit", "http://qudt.org/vocab/unit/")
    graph.bind("sns", str(BASE))
    graph.bind("iao", str(IAO))
    graph.bind("bfo", str(BFO))
    graph.bind("obi", str(OBI))
    graph.bind("ro", str(RO))
    graph.bind("pmdco", str(PMD))
    graph.bind("schema", str(SCHEMA))
    graph.bind("stato", str(STATO))
    return graph


def get_knowledge_graph(
    wf_dict: dict,
    include_t_box: bool = True,
    include_a_box: bool = True,
    hash_data: bool = True,
    remove_data: bool = False,
    extract_dataclasses: bool = False,
    prefix: str | None = None,
) -> Graph:
    """
    Generate RDF graph from a dictionary containing workflow information

    Args:
        wf_dict (dict): dictionary containing workflow information
        include_t_box (bool): if True, include T-Box information
        include_a_box (bool): if True, include A-Box information
        hash_data (bool): if True, compute and include hash values for data nodes
        remove_data (bool): if True, remove data values after hashing
        extract_dataclasses (bool): if True, extract dataclass information into the graph

    Returns:
        (rdflib.Graph): graph containing workflow information
    """
    G = serialize_and_convert_to_networkx(
        wf_dict, hash_data=hash_data, remove_data=remove_data, prefix=prefix
    )
    graph = _get_bound_graph()
    if include_t_box:
        graph += _nx_to_kg(G, t_box=True)
    if include_a_box:
        graph += _nx_to_kg(G, t_box=False)
    if extract_dataclasses:
        graph += extract_dataclass(
            graph=graph, include_t_box=include_t_box, include_a_box=include_a_box
        )
    return graph


def function_to_knowledge_graph(function: Callable) -> Graph:
    """
    Generate RDF graph from a Python function

    Converts a Python function into a knowledge graph representation using RDF.
    The graph includes information about the function's inputs, outputs, and metadata.

    Args:
        function (Callable): Python function to convert into a knowledge graph

    Returns:
        (rdflib.Graph): graph containing the function's semantic representation
    """
    output_args = parse_output_args(function)
    if not isinstance(output_args, tuple):
        output_args = (output_args,)
    input_args = []
    for arg, data in parse_input_args(function).items():
        input_args.append({"arg": arg} | data)
    data = get_function_dict(function)
    f_node = BASE["-".join([data["module"], data["qualname"], data["version"]])]
    return _function_to_graph(
        f_node,
        data=data,
        input_args=input_args,
        output_args=output_args,
        uri=data.get("uri"),
    )


def _function_to_graph(
    f_node: URIRef,
    data: dict,
    input_args: list[dict] | tuple[dict, ...],
    output_args: list[dict] | tuple[dict, ...],
    uri: URIRef | None = None,
) -> Graph:
    """
    Converts a function's metadata into an RDF graph representation.

    Args:
        f_node (URIRef): The URI reference for the function node.
        data (dict): A dictionary containing metadata about the function.
                     Expected keys:
                     - "qualname" (str): The qualified name of the function.
                     - "docstring" (str, optional): The docstring of the function.
        input_args (list[dict]): A list of dictionaries representing input arguments.
        output_args (list[dict]): A list of dictionaries representing output arguments.
        uri (URIRef | None, optional): The URI of the function, if available.

    Returns:
        Graph: An RDF graph representing the function and its metadata.
    """
    g = _get_bound_graph()
    g.add((f_node, RDF.type, SNS.software_method))
    g.add((f_node, RDFS.label, Literal(data["qualname"])))
    if data.get("docstring", "") != "":
        docstring = BNode(f_node + "_docstring")
        g.add((docstring, RDF.type, SNS.textual_entity))
        g.add((docstring, RDF.value, Literal(data["docstring"])))
        g.add((f_node, SNS.denoted_by, docstring))
    if uri is not None:
        g.add((f_node, SNS.is_about, uri))
    if data.get("hash", "") != "":
        hash_bnode = BNode(f_node + "_hash")
        g.add((f_node, SNS.denoted_by, hash_bnode))
        g.add((hash_bnode, RDF.type, SNS.identifier))
        g.add((hash_bnode, RDF.value, Literal(data["hash"])))
    for io, io_args in zip(["input", "output"], [input_args, output_args]):
        for ii, arg in enumerate(io_args):
            if "label" in arg:
                arg_name = arg["label"]
            elif "arg" in arg:
                arg_name = arg["arg"]
            else:
                arg_name = f"output_{ii}"
            arg_node = BNode("_".join([f_node, io, arg_name]))
            if io == "input":
                g.add((arg_node, RDF.type, SNS.input_specification))
            else:
                g.add((arg_node, RDF.type, SNS.output_specification))
            g.add((arg_node, RDFS.label, Literal(arg_name)))
            g.add((f_node, SNS.has_parameter_specification, arg_node))
            g.add(
                (arg_node, SNS.has_parameter_position, Literal(arg.get("position", ii)))
            )
            if "default" in arg:
                g.add(
                    (arg_node, SNS.has_default_literal_value, Literal(arg["default"]))
                )
            if "uri" in arg:
                g.add((arg_node, SNS.is_about, arg["uri"]))
            if "restrictions" in arg:
                g += _restrictions_to_triples(
                    arg["restrictions"],
                    data_node=arg_node,
                    predicate=SNS.has_constraint,
                )
    return g


def _wf_node_to_graph(
    node_name: str,
    data: dict,
    G: SemantikonDiGraph,
    t_box: bool,
) -> Graph:
    g = _get_bound_graph()
    if "function" in data:
        f_node = BASE[data["function"]["identifier"].replace(".", "-")]
        if list(g.triples((f_node, None, None))) == [] and t_box:
            g += _function_to_graph(
                f_node,
                data["function"],
                input_args=[G.nodes[item] for item in G.predecessors(node_name)],
                output_args=[G.nodes[item] for item in G.successors(node_name)],
                uri=data.get("uri"),
            )
    if t_box:
        node = G.t_ns[node_name]
        for io in [G.predecessors(node_name), G.successors(node_name)]:
            for item in io:
                g += _to_owl_restriction(
                    node,
                    SNS.has_part,
                    G.t_ns[item],
                )
        g.add((G.t_ns[node_name], RDFS.subClassOf, SNS.process))
        if "function" in data:
            g += _to_owl_restriction(
                node,
                SNS.has_participant,
                f_node,
                restriction_type=OWL.hasValue,
            )
        g.add((node, RDFS.label, Literal(node_name)))
    else:
        node = G.get_a_node(node_name)
        g.add((node, RDF.type, G.t_ns[node_name]))
        for inp in G.predecessors(node_name):
            g.add((node, SNS.has_part, G.get_a_node(inp)))
        for out in G.successors(node_name):
            g.add((node, SNS.has_part, G.get_a_node(out)))
        if "function" in data:
            g.add((node, SNS.has_participant, f_node))
    return g


def _input_is_connected(io: str, G: SemantikonDiGraph) -> bool:
    candidate = list(G.predecessors(io))
    if len(candidate) == 1:
        if G.nodes[candidate[0]]["step"] == "node":
            return True
        return _input_is_connected(candidate[0], G)
    assert len(candidate) == 0
    return False


def _detect_io_from_str(G: SemantikonDiGraph, seeked_io: str, ref_io: str) -> str:
    assert seeked_io.startswith("inputs") or seeked_io.startswith("outputs")
    main_node = ref_io.replace(".", "-").split("-outputs-")[0].split("-inputs-")[0]
    candidate = (
        G.predecessors(main_node) if "inputs" in seeked_io else G.successors(main_node)
    )
    for io in candidate:
        if io.endswith(seeked_io.replace(".", "-")):
            return G._get_data_node(io=io)
    raise ValueError(f"IO {seeked_io} not found in graph")


def _translate_triples(
    triples: _triple_type,
    node_name: str,
    data_node: BNode,
    G: SemantikonDiGraph,
    t_box: bool,
) -> Graph:
    def _local_str_to_uriref(t: URIRef | BNode | str | None) -> IdentifiedNode | BNode:
        if isinstance(t, SemantikonURI):
            return t.get_instance() if not t_box else t.get_class()
        elif isinstance(t, (URIRef, BNode)):
            return t
        elif t == "self" or t is None:
            return data_node
        else:
            assert isinstance(t, str)
            io = _detect_io_from_str(G=G, seeked_io=t, ref_io=node_name)
            return G.t_ns[io] if t_box else G.get_a_node(io)

    g = _get_bound_graph()
    for triple in triples:
        if len(triple) == 2:
            s = data_node
            p, o = triple
        else:
            s, p, o = triple
        s_n = _local_str_to_uriref(s)
        o_n = _local_str_to_uriref(o)
        if t_box:
            g += _to_owl_restriction(s_n, p, o_n)
        else:
            g.add((s_n, p, o_n))
            for t in [s, o]:
                if isinstance(t, SemantikonURI):
                    g.add((t.get_instance(), RDF.type, t.get_class()))
    return g


def _restrictions_to_triples(
    restrictions: _rest_type, data_node: URIRef | BNode, predicate: URIRef | None = None
) -> Graph:
    """
    Converts restrictions into triples for OWL restrictions or SHACL constraints.

    Args:
        restrictions (_rest_type): The restrictions to convert.
        data_node (URIRef | BNode): The node to which the restrictions apply.
        predicate (URIRef | None): The predicate to use for OWL restrictions
            (default: RDFS.subClassOf).

    Returns:
        Graph: An RDF graph containing the generated triples.
    """
    g = _get_bound_graph()
    assert isinstance(restrictions, tuple | list)
    assert isinstance(restrictions[0], tuple | list)
    if not isinstance(restrictions[0][0], tuple | list):
        restrictions = cast(_rest_type, (restrictions,))

    for r_set in restrictions:
        # Determine whether the restriction is OWL or SHACL based on the predicates
        is_owl = any(r[0] == OWL.onProperty for r in r_set)
        is_shacl = any(r[0] == SH.path for r in r_set)

        assert (
            is_owl ^ is_shacl
        ), "Unable to determine whether the restrictions are OWL or SHACL."
        if is_owl:
            # Create an OWL Restriction
            if predicate is None:
                predicate = RDFS.subClassOf
            b_node = BNode("rest_" + sha256(str(r_set).encode("utf-8")).hexdigest())
            g.add((data_node, predicate, b_node))
            g.add((b_node, RDF.type, OWL.Restriction))
            for r in r_set:
                g.add((b_node, r[0], r[1]))
        elif is_shacl:
            # Create a SHACL NodeShape
            shape_node = BNode(
                "shape_" + sha256(str(r_set).encode("utf-8")).hexdigest()
            )
            if predicate == SNS.has_constraint:
                g.add((data_node, predicate, shape_node))
            else:
                g.add((shape_node, SH.targetClass, data_node))
            g.add((shape_node, RDF.type, SH.NodeShape))
            sh_property = BNode(str(shape_node) + "_property")
            g.add((shape_node, SH.property, sh_property))

            for r in r_set:
                g.add((sh_property, r[0], r[1]))

    return g


def _wf_input_to_graph(
    node_name: str,
    data: dict,
    G: SemantikonDiGraph,
    t_box: bool,
) -> Graph:
    g = _get_bound_graph()
    if t_box:
        data_node = G.t_ns[G._get_data_node(io=node_name)]
        if _input_is_connected(node_name, G):
            out = list(G.predecessors(node_name))
            assert len(out) <= 1
            if len(out) == 1:
                assert G.nodes[out[0]]["step"] in ["outputs", "inputs"]
                if G.nodes[out[0]]["step"] == "outputs":
                    g += _to_owl_restriction(
                        G.t_ns[out[0]], SNS.has_specified_output, data_node
                    )
            if "units" in data:
                g += _to_owl_restriction(
                    base_node=data_node,
                    on_property=QUDT.hasUnit,
                    target_class=_units_to_uri(data["units"]),
                    restriction_type=OWL.hasValue,
                )
            if "uri" in data:
                g += _to_owl_restriction(
                    data_node,
                    SNS.specifies_value_of,
                    data["uri"],
                    restriction_type=OWL.allValuesFrom,
                )
        if "restrictions" in data:
            g += _restrictions_to_triples(data["restrictions"], data_node=data_node)
    else:
        data_node = G.get_a_node(G._get_data_node(io=node_name))
        if not _input_is_connected(node_name, G):
            if "units" in data:
                g.add((data_node, QUDT.hasUnit, _units_to_uri(data["units"])))
            if "uri" in data:
                bnode = BNode(str(data_node) + "_uri")
                g.add((bnode, RDF.type, data["uri"]))
                g.add((data_node, SNS.specifies_value_of, bnode))
    g += _wf_io_to_graph(
        node_name=node_name,
        data=data,
        data_node=data_node,
        G=G,
        io_assignment=SNS.input_assignment,
        has_specified_io=SNS.has_specified_input,
        t_box=t_box,
    )
    return g


def _wf_output_to_graph(
    node_name: str,
    data: dict,
    G: SemantikonDiGraph,
    t_box: bool,
) -> Graph:
    g = _get_bound_graph()
    if t_box:
        data_node = G.t_ns[G._get_data_node(io=node_name)]
    else:
        data_node = G.get_a_node(G._get_data_node(io=node_name))
        if "units" in data:
            g.add((data_node, QUDT.hasUnit, _units_to_uri(data["units"])))
        if "uri" in data:
            bnode = BNode()
            g.add((bnode, RDF.type, data["uri"]))
            g.add((data_node, SNS.specifies_value_of, bnode))
    g += _wf_io_to_graph(
        node_name=node_name,
        data=data,
        data_node=data_node,
        G=G,
        io_assignment=SNS.output_assignment,
        has_specified_io=SNS.has_specified_output,
        t_box=t_box,
    )
    return g


def _wf_io_to_graph(
    node_name: str,
    data: dict,
    data_node: BNode | URIRef,
    G: SemantikonDiGraph,
    io_assignment: URIRef,
    has_specified_io: URIRef,
    t_box: bool,
) -> Graph:
    node = G.t_ns[node_name] if t_box else G.get_a_node(node_name)
    g = _get_bound_graph()
    g.add((node, RDFS.label, Literal(node_name)))
    if t_box:
        g += _to_owl_restriction(node, has_specified_io, data_node)
        g.add((node, RDFS.subClassOf, io_assignment))
        g.add((data_node, RDFS.subClassOf, SNS.value_specification))
    else:
        g.add((data_node, RDF.type, G.t_ns[G._get_data_node(io=node_name)]))
        g.add((node, has_specified_io, data_node))
        if "value" in data and list(g.objects(data_node, RDF.value)) == []:
            g.add((data_node, RDF.value, Literal(data["value"])))
        if "hash" in data:
            hash_bnode = G.get_a_node(G._get_data_node(io=node_name) + "_hash")
            g.add((data_node, SNS.denoted_by, hash_bnode))
            g.add((hash_bnode, RDF.type, SNS.identifier))
            g.add((hash_bnode, RDF.value, Literal(data["hash"])))
    triples = data.get("triples", [])
    if triples != [] and not isinstance(triples[0], list | tuple):
        triples = [triples]
    if "derived_from" in data:
        triples.append(("self", SNS.derives_from, data["derived_from"]))
    if len(triples) > 0:
        g += _translate_triples(
            triples=triples,
            node_name=node_name,
            data_node=data_node,
            G=G,
            t_box=t_box,
        )
    return g


def _parse_precedes(
    G: SemantikonDiGraph,
    workflow_node: URIRef | BNode,
    t_box: bool,
) -> Graph:
    g = _get_bound_graph()
    for node in G.nodes.data():
        if node[1]["step"] == "node":
            successors = list(_get_successor_nodes(G, node[0]))
            if len(successors) == 0:
                if t_box:
                    g += _to_owl_restriction(
                        workflow_node, SNS.has_part, G.t_ns[node[0]]
                    )
                else:
                    g.add((workflow_node, SNS.has_part, G.get_a_node(node[0])))
            else:
                if t_box:
                    for succ in successors:
                        g += _to_owl_restriction(
                            G.t_ns[node[0]],
                            SNS.precedes,
                            G.t_ns[succ],
                        )
                    g += _to_owl_restriction(
                        workflow_node,
                        SNS.has_part,
                        G.t_ns[node[0]],
                    )
                else:
                    for succ in successors:
                        g.add(
                            (
                                G.get_a_node(node[0]),
                                SNS.precedes,
                                G.get_a_node(succ),
                            )
                        )
                    g.add((workflow_node, SNS.has_part, G.get_a_node(node[0])))
    return g


def _parse_global_io(
    G: SemantikonDiGraph,
    workflow_node: URIRef | BNode,
    t_box: bool,
) -> Graph:
    g = _get_bound_graph()
    global_inputs = [
        n for n in G.nodes.data() if G.in_degree(n[0]) == 0 and n[1]["step"] == "inputs"
    ]
    global_outputs = [
        n
        for n in G.nodes.data()
        if G.out_degree(n[0]) == 0 and n[1]["step"] == "outputs"
    ]
    for global_io in [global_inputs, global_outputs]:
        for io in global_io:
            if t_box:
                g += _to_owl_restriction(
                    workflow_node,
                    SNS.has_part,
                    G.t_ns[io[0]],
                )
            else:
                g.add((workflow_node, SNS.has_part, G.get_a_node(io[0])))
    return g


def _nx_to_kg(G: SemantikonDiGraph, t_box: bool) -> Graph:
    g = _get_bound_graph()
    workflow_node = G.t_ns[G.name] if t_box else G.get_a_node(G.name)
    for node_name, data in G.nodes.data():
        data = data.copy()
        step = data.pop("step")
        if t_box:
            g.add((G.t_ns[node_name], RDF.type, OWL.Class))
        else:
            g.add((G.get_a_node(node_name), RDF.type, G.t_ns[node_name]))
        assert step in ["node", "inputs", "outputs"], f"Unknown step: {step}"
        if step == "node":
            g += _wf_node_to_graph(
                node_name=node_name,
                data=data,
                G=G,
                t_box=t_box,
            )
        elif step == "inputs":
            g += _wf_input_to_graph(
                node_name=node_name,
                data=data,
                G=G,
                t_box=t_box,
            )
        else:
            g += _wf_output_to_graph(
                node_name=node_name,
                data=data,
                G=G,
                t_box=t_box,
            )

    if t_box:
        g.add((workflow_node, RDF.type, OWL.Class))
        g.add((workflow_node, RDFS.subClassOf, SNS.process))
        g.add((workflow_node, RDFS.label, Literal(G.name)))
    else:
        g.add((workflow_node, RDF.type, G.t_ns[G.name]))
    g += _parse_precedes(G=G, workflow_node=workflow_node, t_box=t_box)
    g += _parse_global_io(G=G, workflow_node=workflow_node, t_box=t_box)
    return g


class _DataclassTranslator:
    """
    Internal helper class responsible for translating Python dataclasses
    into RDF TBox and ABox statements.

    This class encapsulates the recursive traversal logic and graph
    construction in order to reduce cyclomatic complexity of the public
    API functions.
    """

    def __init__(
        self,
        *,
        include_t_box: bool = True,
        include_a_box: bool = True,
    ) -> None:
        """
        Initialize the translator.

        Args:
            include_t_box: Whether to emit TBox axioms (classes, restrictions).
            include_a_box: Whether to emit ABox assertions (individuals, values).
        """
        self.include_t_box = include_t_box
        self.include_a_box = include_a_box

    def translate(
        self,
        *,
        a_node: BNode,
        t_node: URIRef,
        value: Any,
        dtype: type,
    ) -> Graph:
        """
        Translate a single dataclass instance and its type into RDF.

        Args:
            a_node: ABox node representing the dataclass instance.
            t_node: TBox node representing the dataclass type.
            value: The Python dataclass instance.
            dtype: The dataclass type.

        Returns:
            An RDF graph containing the generated triples.
        """
        g = _get_bound_graph()

        self._translate_nested_dataclasses(
            graph=g,
            a_node=a_node,
            t_node=t_node,
            value=value,
            dtype=dtype,
        )

        for field, annotation in dtype.__annotations__.items():
            metadata = meta_to_dict(annotation)

            t_field = self._to_subkey(t_node, field)
            a_field = self._to_subkey(a_node, field)
            field_value = getattr(value, field, None)

            if self.include_t_box:
                self._emit_tbox(
                    graph=g,
                    parent=t_node,
                    field_node=t_field,
                    metadata=metadata,
                )

            if self.include_a_box:
                self._emit_abox(
                    graph=g,
                    field_node=a_field,
                    field_class=t_field,
                    metadata=metadata,
                    value=field_value,
                )

        return g

    def _to_subkey(self, node: URIRef | BNode, key: str):
        """
        Construct a deterministic sub-node for a dataclass field.

        Args:
            node: Base URIRef or BNode.
            key: Field name.

        Returns:
            A new URIRef or BNode derived from the base node.
        """
        base = str(node).rsplit("_data", 1)[0]
        return node.__class__(f"{base}_{key}_data")

    def _translate_nested_dataclasses(
        self,
        *,
        graph: Graph,
        a_node: BNode,
        t_node: URIRef,
        value: Any,
        dtype: type,
    ) -> None:
        """
        Recursively translate nested dataclass fields.

        Args:
            graph: Graph to populate.
            a_node: Parent ABox node.
            t_node: Parent TBox node.
            value: Dataclass instance.
            dtype: Dataclass type.
        """
        for field, field_type in dtype.__dict__.items():
            if isinstance(field_type, type) and is_dataclass(field_type):
                graph += self.translate(
                    a_node=self._to_subkey(a_node, field),
                    t_node=self._to_subkey(t_node, field),
                    value=getattr(value, field, None),
                    dtype=field_type,
                )

    def _emit_tbox(
        self,
        *,
        graph: Graph,
        parent: URIRef,
        field_node: URIRef,
        metadata: dict,
    ) -> None:
        """
        Emit TBox axioms for a dataclass field.

        Args:
            graph: Graph to populate.
            parent: Parent class.
            field_node: Field class.
            metadata: Parsed annotation metadata.
        """
        graph.add((field_node, RDFS.subClassOf, parent))

        if "units" in metadata:
            graph += _to_owl_restriction(
                base_node=field_node,
                on_property=QUDT.hasUnit,
                target_class=_units_to_uri(metadata["units"]),
                restriction_type=OWL.hasValue,
            )

        if "uri" in metadata:
            graph += _to_owl_restriction(
                base_node=field_node,
                on_property=SNS.specifies_value_of,
                target_class=metadata["uri"],
            )

    def _emit_abox(
        self,
        *,
        graph: Graph,
        field_node: BNode,
        field_class: URIRef,
        metadata: dict,
        value: Any,
    ) -> None:
        """
        Emit ABox assertions for a dataclass field.

        Args:
            graph: Graph to populate.
            field_node: Individual representing the field value.
            field_class: Class of the field.
            metadata: Parsed annotation metadata.
            value: Python value of the field.
        """
        graph.add((field_node, RDF.type, field_class))

        if "units" in metadata:
            graph.add((field_node, QUDT.hasUnit, _units_to_uri(metadata["units"])))

        if "uri" in metadata:
            bnode = BNode()
            graph.add((bnode, RDF.type, metadata["uri"]))
            graph.add((field_node, SNS.specifies_value_of, bnode))

        if value is not None:
            graph.add((field_node, RDF.value, Literal(value)))


def extract_dataclass(
    graph: Graph,
    include_t_box: bool = True,
    include_a_box: bool = True,
) -> Graph:
    """
    Extract dataclass-backed RDF.value entries from a graph and translate
    them into RDF TBox and/or ABox triples.

    This function preserves the original public API while delegating
    the translation logic to an internal helper class.

    Args:
        graph: Input RDF graph.
        include_t_box: Whether to include TBox axioms.
        include_a_box: Whether to include ABox assertions.

    Returns:
        A new RDF graph containing the extracted triples.
    """
    out = _get_bound_graph()
    translator = _DataclassTranslator(
        include_t_box=include_t_box,
        include_a_box=include_a_box,
    )

    for subj, obj in graph.subject_objects(RDF.value):
        py_value = obj.toPython()
        if not is_dataclass(py_value):
            continue

        t_nodes = list(graph.objects(subj, RDF.type))
        assert len(t_nodes) == 1

        out += translator.translate(
            a_node=subj,
            t_node=t_nodes[0],
            value=py_value,
            dtype=type(py_value),
        )

    return out


def _get_successor_nodes(G, node_name):
    for out in G.successors(node_name):
        for inp in G.successors(out):
            for node in G.successors(inp):
                yield node


class _WorkflowGraphSerializer:
    """
    Serializes a workflow dictionary into a SemantikonDiGraph.
    """

    def __init__(self, wf_dict: dict, prefix: str | None = None):
        self.wf_dict = wf_dict
        self.node_dict: dict = {}
        self.channel_dict: dict = {}
        self.edge_list: list[list[str]] = []
        self.prefix = prefix

    # -----------------------------
    # String utilities
    # -----------------------------

    @staticmethod
    def _remove_us(*args: str) -> str:
        s = ".".join(args)
        return ".".join(part.split("__")[-1] for part in s.split("."))

    @staticmethod
    def _dot(*args: str | None) -> str:
        return ".".join(a for a in args if a is not None)

    # -----------------------------
    # Serialization
    # -----------------------------

    def serialize(self) -> SemantikonDiGraph:
        self._serialize_workflow(self.wf_dict, self.wf_dict["label"])
        return self._build_graph()

    def _serialize_workflow(self, wf_dict: dict, prefix: str) -> None:
        self._serialize_node_metadata(wf_dict, prefix)
        self._serialize_channels(wf_dict, prefix)
        self._serialize_children(wf_dict, prefix)
        self._serialize_edges(wf_dict, prefix)

    def _serialize_node_metadata(self, wf_dict: dict, prefix: str) -> None:
        self.node_dict[prefix] = {
            k: v for k, v in wf_dict.items() if k not in {"nodes", "edges"}
        }

        assert "function" in wf_dict or wf_dict["type"] != "Function"

        if "function" in wf_dict:
            meta = get_function_dict(wf_dict["function"])
            meta["identifier"] = ".".join(
                (meta["module"], meta["qualname"], meta["version"])
            )
            self.node_dict[prefix]["function"] = meta

    def _serialize_channels(self, wf_dict: dict, prefix: str) -> None:
        for io_type in ("inputs", "outputs"):
            for pos, (arg, channel) in enumerate(wf_dict.get(io_type, {}).items()):
                label = self._remove_us(prefix, io_type, arg)
                assert "semantikon_type" not in channel

                self.channel_dict[label] = channel | {
                    "semantikon_type": io_type,
                    "position": channel.get("position", pos),
                    "arg": arg,
                }

    def _serialize_children(self, wf_dict: dict, prefix: str) -> None:
        for key, node in wf_dict.get("nodes", {}).items():
            self._serialize_workflow(node, self._dot(prefix, key))

    def _serialize_edges(self, wf_dict: dict, prefix: str) -> None:
        for edge in wf_dict.get("edges", []):
            self.edge_list.append([self._remove_us(prefix, a) for a in edge])

    # -----------------------------
    # Graph construction
    # -----------------------------

    def _build_graph(self) -> SemantikonDiGraph:
        G = SemantikonDiGraph(prefix=self.prefix)

        self._add_channels(G)
        self._add_nodes(G)
        self._add_edges(G)

        return self._relabel_graph(G)

    def _add_channels(self, G: SemantikonDiGraph) -> None:
        for key, data in self.channel_dict.items():
            G.add_node(
                key,
                step=data["semantikon_type"],
                **{k: v for k, v in data.items() if k != "semantikon_type"},
            )

    def _add_nodes(self, G: SemantikonDiGraph) -> None:
        for key, data in self.node_dict.items():
            if "." not in key:
                G.name = key
                continue

            G.add_node(
                key,
                step="node",
                **{k: v for k, v in data.items() if k not in {"inputs", "outputs"}},
            )

            for inp in data.get("inputs", {}):
                G.add_edge(f"{key}.inputs.{inp}", key)

            for out in data.get("outputs", {}):
                G.add_edge(key, f"{key}.outputs.{out}")

    def _add_edges(self, G: SemantikonDiGraph) -> None:
        for edge in self.edge_list:
            G.add_edge(*edge)

    @staticmethod
    def _relabel_graph(G: SemantikonDiGraph) -> SemantikonDiGraph:
        mapping = {n: n.replace(".", "-") for n in G.nodes()}
        return nx.relabel_nodes(G, mapping, copy=True)


def serialize_and_convert_to_networkx(
    wf_dict: dict,
    hash_data: bool = True,
    remove_data: bool = False,
    prefix: str | None = None,
) -> SemantikonDiGraph:
    """
    Serialize a workflow dictionary into a SemantikonDiGraph, optionally
    hashing node data.

    Args:
        wf_dict (dict): The workflow dictionary to serialize.
        hash_data (bool): Whether to hash node data.
        remove_data (bool): Whether to remove original data after hashing.

    Returns:
        SemantikonDiGraph: The serialized workflow graph.
    """
    G = _WorkflowGraphSerializer(wf_dict, prefix=prefix).serialize()
    if hash_data:
        hashed_dict = get_hashed_node_dict(wf_dict)
        for node, data in hashed_dict.items():
            G.append_hash(node, data["hash"], remove_data=remove_data)
    return G


def _to_owl_restriction(
    base_node: URIRef | None,
    on_property: URIRef,
    target_class: URIRef,
    restriction_type: URIRef = OWL.someValuesFrom,
) -> Graph:
    g = _get_bound_graph()
    restriction_node = BNode(
        sha256(
            str((base_node, on_property, target_class, restriction_type)).encode(
                "utf-8"
            )
        ).hexdigest()
    )

    # Build the restriction
    g.add((restriction_node, RDF.type, OWL.Restriction))
    g.add((restriction_node, OWL.onProperty, on_property))
    g.add((restriction_node, restriction_type, target_class))
    if base_node is not None:
        g.add((base_node, RDFS.subClassOf, restriction_node))

    return g


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
    G_tmp = G.copy()
    for node in G_tmp.nodes:
        if "default" in G_tmp.nodes[node]:
            default = G_tmp.nodes[node].pop("default")
            if "value" not in G_tmp.nodes[node]:
                G_tmp.nodes[node]["value"] = default
        if "value" in G_tmp.nodes[node]:
            if G_tmp.in_degree(node) > 0 or not with_global_inputs:
                del G_tmp.nodes[node]["value"]
            elif is_dataclass(G_tmp.nodes[node]["value"]):
                G_tmp.nodes[node]["value"] = asdict(G_tmp.nodes[node]["value"])
        if "dtype" in G_tmp.nodes[node]:
            del G_tmp.nodes[node]["dtype"]

    for _, attrs in G_tmp.nodes(data=True):
        attrs["canon"] = json.dumps(attrs, sort_keys=True)

    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
        G_tmp, node_attr="canon"
    )


def _get_undefined_connections(g, term):
    query = "\n".join(
        [f"PREFIX {key}: <{value}>" for key, value in dict(g.namespaces()).items()]
    )
    query += f"""
    SELECT ?rnode WHERE {{
      ?class rdfs:subClassOf ?rnode .
      ?rnode a owl:Restriction .
      ?rnode owl:onProperty {term} .
      ?instance a ?class .
      FILTER NOT EXISTS {{
        ?instance {term} ?bnode .
      }}
    }}"""
    return [item for items in g.query(query) for item in items]


class _OWLToSHACLConverter:
    def __init__(self, owl_graph: Graph, excluded_nodes: list[BNode] | None = None):
        """
        Initialize the converter with an OWL graph and optional excluded nodes.
        """
        self.owl_graph = owl_graph
        self.excluded_nodes = excluded_nodes
        self.shacl_graph = _get_bound_graph()
        self.node_shapes: dict[URIRef, BNode] = {}
        self.shacl_graph.bind("sh", str(SH))
        self.shacl_graph.bind("sns", str(BASE))

    def iter_supported_restrictions(self):
        """
        Yield (base_class, restriction_node, property, restriction_type, value)
        for supported OWL restrictions.
        """
        for r in self.owl_graph.subjects(RDF.type, OWL.Restriction):
            if self.excluded_nodes is not None and r in self.excluded_nodes:
                continue
            prop = self.owl_graph.value(r, OWL.onProperty)
            if prop is None:
                continue
            for restriction_type in (
                OWL.someValuesFrom,
                OWL.hasValue,
                OWL.allValuesFrom,
            ):
                value = self.owl_graph.value(r, restriction_type)
                if value is None:
                    continue

                for base_cls in self.owl_graph.subjects(RDFS.subClassOf, r):
                    yield base_cls, prop, restriction_type, value

    def convert(self) -> Graph:
        """
        Convert the OWL restrictions in the graph to SHACL shapes.
        """
        for base_cls, prop, rtype, value in self.iter_supported_restrictions():

            # One NodeShape per base class
            if base_cls not in self.node_shapes:
                ns = BNode()
                self.node_shapes[base_cls] = ns
                self.shacl_graph.add((ns, RDF.type, SH.NodeShape))
                self.shacl_graph.add((ns, SH.targetClass, base_cls))
            else:
                ns = self.node_shapes[base_cls]

            ps = BNode()
            self.shacl_graph.add((ps, RDF.type, SH.PropertyShape))
            self.shacl_graph.add((ps, SH.path, prop))

            if rtype == OWL.someValuesFrom:
                # Existential restriction:
                # ∃ prop . C  →  qualifiedValueShape + qualifiedMinCount
                qvs = BNode()
                self.shacl_graph.add((qvs, SH["class"], value))

                self.shacl_graph.add((ps, SH.qualifiedValueShape, qvs))
                self.shacl_graph.add((ps, SH.qualifiedMinCount, Literal(1)))

            elif rtype == OWL.hasValue:
                self.shacl_graph.add((ps, SH.hasValue, value))

            elif rtype == OWL.allValuesFrom:
                # Universal restriction:
                # ∀ prop . C  →  class
                self.shacl_graph.add((ps, SH["class"], value))

            self.shacl_graph.add((ns, SH.property, ps))

        return self.shacl_graph


def owl_restrictions_to_shacl(
    owl_graph: Graph, excluded_nodes: list[BNode] | None = None
) -> Graph:
    """
    Convert OWL restrictions in the given graph to SHACL shapes.

    This function is a wrapper around the _OWLToSHACLConverter class.
    """
    converter = _OWLToSHACLConverter(owl_graph, excluded_nodes)
    return converter.convert()


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

    def query(self) -> list[list[Any]]:
        qn = _QueryHolder([self], self._graph)
        return qn.query()

    def to_query_text(self) -> str:
        qn = _QueryHolder([self], self._graph)
        return qn.to_query_text()

    def value(self) -> URIRef:
        return BASE["-".join(self._path)]

    def __add__(self, other) -> _QueryHolder:
        if isinstance(other, _Node):
            nodes = [self, other]
        else:
            assert isinstance(other, _QueryHolder)
            nodes = [self] + other._nodes
        return _QueryHolder(nodes, self._graph)


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

    def to_query_graph(self):
        """Build the intermediate query graph for the held nodes.

        The query graph is created by delegating to :class:`SparqlWriter`,
        which inspects the provided RDF graph and the stored nodes.

        Returns
        -------
        Graph
            An RDFLib :class:`Graph` representing the SPARQL query structure.
        """
        sw = SparqlWriter(self._graph)
        return sw.get_query_graph(*self._nodes)

    def to_query_text(self):
        """Generate a SPARQL query string for the held nodes.

        The method first builds the intermediate query graph via
        :meth:`to_query_graph` and then converts it to a textual SPARQL
        representation.

        Returns
        -------
        str
            A SPARQL query string that can be executed against the stored
            RDFLib :class:`Graph`.
        """
        G = self.to_query_graph()
        return SparqlWriter.get_query_text(G)

    def query(self):
        """Execute the generated SPARQL query against the stored graph.

        The query text produced by :meth:`to_query_text` is run with
        :meth:`Graph.query`, and each bound value in the result set is
        converted to a native Python object via its ``toPython`` method.

        Returns
        -------
        list[list[Any]]
            A list of result rows, where each row is a list of converted
            Python values corresponding to the query variables.
        """
        text = self.to_query_text()
        return [[a.toPython() for a in item] for item in self._graph.query(text)]

    def __add__(self, other) -> _QueryHolder:
        if isinstance(other, _Node):
            nodes = self._nodes + [other]
        else:
            assert isinstance(other, _QueryHolder)
            nodes = self._nodes + other._nodes
        return _QueryHolder(nodes, self._graph)


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
        candidates = list(self._graph.subjects(RDFS.subClassOf, SNS.process))
        for node in nx.topological_sort(self.G):
            if node in candidates and all(
                nx.has_path(self.G, node, dn) for dn in data_nodes
            ):
                return node
        raise ValueError("No common head node found")

    def get_query_graph(self, *args) -> nx.DiGraph:
        """
        Generate a query graph based on the provided arguments.

        The query graph is a directed graph (DiGraph) where nodes represent
        data elements and edges represent relationships between them. This graph
        can be used to generate SPARQL query text.

        Args:
            *args: A variable number of arguments representing nodes in the graph.
                   Each argument can be an RDFLib node or a value.

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
            G.add_node(self._to_qname(data_nodes[-1] + "_value"), output=ii)
            G.add_node(data_nodes[-1])
            G.add_edge(self._to_qname(data_nodes[-1]), data_nodes[-1], predicate="a")
            G.add_edge(
                self._to_qname(data_nodes[-1]),
                self._to_qname(data_nodes[-1] + "_value"),
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
        output_with_ind = [
            [data["output"], f"?{to_identifier(node)}"]
            for node, data in G.nodes.data()
            if "output" in data
        ]
        output_args = [x for _, x in sorted(output_with_ind, key=lambda pair: pair[0])]
        lines = ["SELECT " + " ".join(output_args) + " WHERE {"]
        for subj, obj, data in G.edges.data():
            subj, obj = [
                f"<{e}>" if isinstance(e, URIRef) else f"?{to_identifier(e)}"
                for e in [subj, obj]
            ]
            pred = (
                f"<{data['predicate']}>"
                if isinstance(data["predicate"], URIRef)
                else data["predicate"]
            )
            lines.append(f"{subj} {pred} {obj} .")
        lines.append("}")
        return "\n".join(lines)


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


def label_to_uri(graph: Graph, label: str | URIRef) -> list[URIRef]:
    """
    Convert a human-readable label to its corresponding URIRef in the graph.

    Args:
        graph (Graph): The RDF graph to query.
        label (str | URIRef): The human-readable label or URIRef.

    Returns:
        str: The corresponding URIRef in the graph.
    """
    if isinstance(label, URIRef) or (isinstance(label, str) and label.startswith("http")):
        label = graph.qname(URIRef(label)).split(":")[-1]
    query = f"""SELECT ?s
    WHERE {{
      ?s rdfs:label ?label .
      ?s a owl:Class .
      FILTER(?label = "{label}")
    }}"""
    result = list(graph.query(query))
    assert len(result) > 0, f"No result found for {label}"
    return [r[0] for r in result]
