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
        return BASE

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
    G = serialize_and_convert_to_networkx(wf_dict)
    if hash_data:
        hashed_dict = get_hashed_node_dict(wf_dict)
        for node, data in hashed_dict.items():
            G.append_hash(node, data["hash"], remove_data=remove_data)
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
        for io in [G.predecessors(node_name), G.successors(node_name)]:
            for item in io:
                g += _to_owl_restriction(
                    G.t_ns[node_name],
                    SNS.has_part,
                    G.t_ns[item],
                )
        g.add((G.t_ns[node_name], RDFS.subClassOf, SNS.process))
        if "function" in data:
            g += _to_owl_restriction(
                G.t_ns[node_name],
                SNS.has_participant,
                f_node,
                restriction_type=OWL.hasValue,
            )
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
    if data.get("label") is not None:
        g.add((node, RDFS.label, Literal(data["label"])))
    else:
        g.add((node, RDFS.label, Literal(node_name.split("-")[-1])))
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
            hash_bnode = BNode(G._get_data_node(io=node_name) + "_hash")
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

    def __init__(self, wf_dict: dict):
        self.wf_dict = wf_dict
        self.node_dict: dict = {}
        self.channel_dict: dict = {}
        self.edge_list: list[list[str]] = []

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
        G = SemantikonDiGraph()

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


def serialize_and_convert_to_networkx(wf_dict: dict) -> SemantikonDiGraph:
    return _WorkflowGraphSerializer(wf_dict).serialize()


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
    __slots__ = ("_node", "_path")

    def __init__(self, node: TrieNode, path: Iterable[str]):
        self._node = node
        self._path = tuple(path)

    def __getattr__(self, name: str):
        if name not in self._node.children:
            raise AttributeError(name)
        return _Node(self._node.children[name], self._path + (name,))

    def __dir__(self):
        return sorted(self._node.children.keys())

    def value(self) -> URIRef:
        return BASE["-".join(self._path)]


class Completer(_Node):
    def __init__(self, values: Iterable[str]):
        root = TrieNode()
        for value in values:
            node = root
            for part in value.split("-"):
                node = node.children.setdefault(part, TrieNode())
            node.terminal = True

        super().__init__(root, ())


def query_io_completer(graph: Graph) -> Completer:
    all_ios = []
    for pred in ["pmd:0000066", "pmd:0000067"]:
        query = f"""
        PREFIX pmd: <https://w3id.org/pmd/co/PMD_>
        SELECT ?io WHERE {{
            ?io rdfs:subClassOf {pred} .
        }}"""
        all_ios.extend([g[0].split("/")[-1] for g in graph.query(query)])
    return Completer(all_ios)


class SparqlWriter:
    def __init__(self, graph: Graph):
        self.graph = graph

    @cached_property
    def G(self) -> nx.DiGraph:
        query = """
        SELECT ?parent ?property ?child WHERE {
            ?parent rdfs:subClassOf ?bnode .
            ?bnode a owl:Restriction .
            ?bnode owl:onProperty ?property .
            ?bnode owl:someValuesFrom ?child .
        }"""
        G = nx.DiGraph()
        for subj, pred, obj in self.graph.query(query):
            G.add_edge(subj, obj, predicate=pred)
        return G

    def get_query_graph(self, *args):
        G = nx.DiGraph()
        data_nodes = []
        for ii, arg in enumerate(args):
            if isinstance(arg, _Node):
                arg = arg.value()
            data_nodes.append(list(self.G.successors(arg))[0])
            G.add_node(BNode(data_nodes[-1] + "_value"), output=True)
            G.add_edge(BNode(data_nodes[-1]), data_nodes[-1], predicate="a")
            G.add_edge(
                BNode(data_nodes[-1]),
                BNode(data_nodes[-1] + "_value"),
                predicate="rdf:value",
            )
        query_text = ""
        if len(data_nodes) > 1:
            for u, v in zip(data_nodes[:-1], data_nodes[1:]):
                paths = nx.shortest_path(self.G.to_undirected(), u, v)
                for uu, vv in zip(paths[:-1], paths[1:]):
                    if self.G.has_edge(uu, vv):
                        G.add_edge(
                            BNode(uu),
                            BNode(vv),
                            predicate=self.G.edges[uu, vv]["predicate"],
                        )
                    else:
                        G.add_edge(
                            BNode(vv),
                            BNode(uu),
                            predicate=self.G.edges[vv, uu]["predicate"],
                        )
        return G

    def get_query_text(self, *args) -> list[list[Any]]:
        G = self.get_query_graph(*args)
        output_args = [
            f"?{to_identifier(node)}"
            for node, data in G.nodes.data()
            if data.get("output", False)
        ]
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

    def query(self, *args):
        return [
            [a.toPython() for a in item]
            for item in self.graph.query(self.get_query_text(*args))
        ]
