import uuid
from dataclasses import dataclass
from typing import Any, TypeAlias

import networkx as nx
from flowrep.tools import get_function_metadata
from owlrl import DeductiveClosure, OWLRL_Semantics
from rdflib import OWL, PROV, RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.term import IdentifiedNode

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
    is_specified_input_of: URIRef = OBI["0000295"]
    has_specified_output: URIRef = OBI["0000299"]
    is_specified_output_of: URIRef = OBI["0000312"]
    has_unit: URIRef = QUDT["hasUnit"]
    input_assignment: URIRef = PMD["0000066"]
    executes: URIRef = STATO["0000102"]
    output_assignment: URIRef = PMD["0000067"]
    precedes: URIRef = BFO["0000063"]
    process: URIRef = BFO["0000015"]
    continuant: URIRef = BFO["0000002"]
    value_specification: URIRef = OBI["0001933"]
    specifies_value_of: URIRef = OBI["0001927"]


class NS:
    PREFIX = "semantikon_parent_prefix"
    TYPE = "semantikon_type"


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


def _check_missing_triples(graph: Graph) -> list:
    return []


def _check_connections(graph: Graph, strict_typing: bool = False, ontology=SNS) -> list:
    """
    Check if the connections between inputs and outputs are compatible

    Args:
        graph (rdflib.Graph): graph to be validated
        strict_typing (bool): if True, check for strict typing

    Returns:
        (list): list of incompatible connections
    """
    return []


def _check_units(graph: Graph, ontology=SNS) -> dict[URIRef, list[URIRef]]:
    """
    Check if there are multiple units assigned to the same term

    Args:
        graph (rdflib.Graph): graph to be validated

    Returns:
        (dict): dictionary of terms with multiple units
    """
    return {}


def validate_values(
    graph: Graph, run_reasoner: bool = True, strict_typing: bool = False, ontology=SNS
) -> dict[str, Any]:
    """
    Validate if all values required by restrictions are present in the graph

    Args:
        graph (rdflib.Graph): graph to be validated
        run_reasoner (bool): if True, run the reasoner
        strict_typing (bool): if True, check for strict typing

    Returns:
        (dict): list of missing triples
    """
    if run_reasoner:
        DeductiveClosure(OWLRL_Semantics).expand(graph)
    return {
        "missing_triples": _check_missing_triples(graph),
        "incompatible_connections": _check_connections(
            graph, strict_typing=strict_typing, ontology=ontology
        ),
        "distinct_units": _check_units(graph, ontology=ontology),
    }


def _remove_us(*arg) -> str:
    s = ".".join(arg)
    return ".".join(t.split("__")[-1] for t in s.split("."))


def _dot(*args) -> str:
    return ".".join([a for a in args if a is not None])


def extract_dataclass(
    graph: Graph, namespace: Namespace | None = None, ontology=SNS
) -> Graph:
    return Graph()


def get_knowledge_graph(
    wf_dict: dict,
    t_box: bool = False,
    graph: Graph | None = None,
    inherit_properties: bool = True,
    ontology=SNS,
    append_missing_items: bool = True,
    use_uuid: bool = False,
    namespace: Namespace | None = None,
) -> Graph:
    """
    Generate RDF graph from a dictionary containing workflow information

    Args:
        wf_dict (dict): dictionary containing workflow information
        t_box (bool): if True, generate T-Box graph, otherwise A-Box graph
        graph (rdflib.Graph): graph to be updated
        inherit_properties (bool): if True, properties are inherited
        ontology (Namespace): ontology to be used
        append_missing_items (bool): if True, append missing items for the
            OWL restrictions
        use_uuid (bool): if True, use UUIDs for node labels

    Returns:
        (rdflib.Graph): graph containing workflow information
    """
    if graph is None:
        graph = Graph()
    node_dict, channel_dict, edge_list = serialize_data(wf_dict, use_uuid=use_uuid)
    G = _wf_data_to_networkx(node_dict, channel_dict, edge_list)
    graph += _nx_to_kg(G, t_box=t_box)
    if namespace is not None:
        graph.bind("ns", str(namespace))
    return graph


def _wf_node_to_graph(
    node_name: str, kg_node: URIRef, G: nx.DiGraph, t_box: bool
) -> Graph:
    g = Graph()
    if t_box:
        g_rest = Graph()
        for inp in G.predecessors(node_name):
            g_rest += _to_owl_restriction(SNS.has_part, BASE[inp])
        for out in G.successors(node_name):
            g_rest += _to_owl_restriction(SNS.has_part, BASE[out])
        for nn in _bundle_restrictions(g_rest):
            g.add((kg_node, RDFS.subClassOf, nn))
        g.add((kg_node, RDFS.subClassOf, SNS.process))
        g += g_rest
    else:
        g.add((kg_node, RDF.type, SNS.process))
        for inp in G.predecessors(node_name):
            g.add((kg_node, SNS.has_part, BASE[inp]))
        for out in G.successors(node_name):
            g.add((kg_node, SNS.has_part, BASE[out]))
    return g


def _wf_io_to_graph(
    step: str,
    node_name: str,
    node: URIRef,
    data: dict,
    G: nx.DiGraph,
    t_box: bool,
) -> Graph:
    g = Graph()
    if data.get("label") is not None:
        g.add(node, RDFS.label, Literal(data["label"]))
    if t_box:
        data_class = SNS.continuant if "uri" not in data else data["uri"]
        data_node = BNode(node + "_data")
        io_assignment = SNS.input_assignment if step == "inputs" else SNS.output_assignment
        has_specified_io = (
            SNS.has_specified_input if step == "inputs" else SNS.has_specified_output
        )
        g_rest = _to_owl_restriction(has_specified_io, data_node)
        rest = _bundle_restrictions(g_rest)
        g += g_rest
        g += _to_intersection(node, [io_assignment] + rest)
        if data.get("units") is not None:
            g_units = _to_owl_restriction(
                SNS.has_unit, _units_to_uri(data["units"]), OWL.hasValue,
            )
        else:
            g_units = Graph()
        if step == "inputs":
            out = list(G.predecessors(node_name))
            if len(out) == 1:
                g_rest = _to_owl_restriction(SNS.is_specified_output_of, BASE[out[0]])
                g_rest += g_units
                rest = _bundle_restrictions(g_rest)
                g += g_rest
                g += _to_intersection(data_node, [data_class] + rest)
            elif len(out) == 0:
                g_rest = _to_owl_restriction(SNS.specifies_value_of, data_class)
                g_rest += g_units
                rest = _bundle_restrictions(g_rest)
                g += g_rest
                g += _to_intersection(data_node, [SNS.value_specification] + rest)
            else:
                raise AssertionError
        elif step == "outputs":
            g_rest = _to_owl_restriction(SNS.specifies_value_of, data_class)
            g_rest += g_units
            rest = _bundle_restrictions(g_rest)
            g += g_rest
            g += _to_intersection(data_node, [SNS.value_specification] + rest)
    else:
        if step == "inputs":
            g.add((node, RDF.type, SNS.input_assignment))
        elif step == "outputs":
            g.add((node, RDF.type, SNS.output_assignment))
    return g


def _nx_to_kg(G: nx.DiGraph, t_box: bool) -> Graph:
    g = Graph()
    g.bind("qudt", str(QUDT))
    g.bind("unit", "http://qudt.org/vocab/unit/")
    g.bind("sns", str(BASE))
    g.bind("prov", str(PROV))
    g.bind("iao", str(IAO))
    g.bind("bfo", str(BFO))
    g.bind("obi", str(OBI))
    g.bind("ro", str(RO))
    g.bind("pmdco", str(PMD))
    g.bind("schema", str(SCHEMA))
    g.bind("stato", str(STATO))
    workflow_name = G.name
    for comp in G.nodes.data():
        data = comp[1].copy()
        step = data.pop("step")
        node = BASE[comp[0]]
        if t_box:
            g.add((node, RDF.type, OWL.Class))
        if step == "node":
            g += _wf_node_to_graph(comp[0], node, G, t_box)
        elif step in ["inputs", "outputs"]:
            g += _wf_io_to_graph(
                step=step, node_name=comp[0], node=node, data=data, G=G, t_box=t_box
            )
        else:
            raise AssertionError

    g_all_nodes = Graph()
    g.add((BASE[workflow_name], RDF.type, OWL.Class))
    for node in G.nodes.data():
        if node[1]["step"] == "node":
            successors = list(_get_successor_nodes(G, node[0]))
            if len(successors) == 0:
                g_all_nodes += _to_owl_restriction(SNS.has_part, BASE[node[0]])
            else:
                g_rest = Graph()
                node_tmp = BNode()
                for succ in successors:
                    g_rest += _to_owl_restriction(SNS.precedes, BASE[succ])
                g_all_nodes += g_rest
                rest = _bundle_restrictions(g_rest)
                g_all_nodes += _to_intersection(node_tmp, [BASE[node[0]]] + rest)
                g_all_nodes += _to_owl_restriction(SNS.has_part, node_tmp)
    for nn in _bundle_restrictions(g_all_nodes):
        g.add((BASE[workflow_name], OWL.equivalentClass, nn))
    g += g_all_nodes

    global_inputs = [
        n for n in G.nodes.data() if G.in_degree(n[0]) == 0 and n[1]["step"] == "inputs"
    ]
    global_outputs = [
        n
        for n in G.nodes.data()
        if G.out_degree(n[0]) == 0 and n[1]["step"] == "outputs"
    ]
    g_io_nodes = Graph()
    for inp in global_inputs:
        g_io_nodes += _to_owl_restriction(SNS.has_specified_input, BASE[inp[0]])
    for out in global_outputs:
        g_io_nodes += _to_owl_restriction(SNS.has_specified_output, BASE[inp[0]])
    g += g_io_nodes
    for nn in _bundle_restrictions(g_io_nodes):
        g.add((BASE[workflow_name], RDFS.subClassOf, nn))
    g.add((BASE[workflow_name], RDFS.subClassOf, SNS.process))
    return g


def _get_successor_nodes(G, node_name):
    for out in G.successors(node_name):
        for inp in G.successors(out):
            for node in G.successors(inp):
                yield node


def serialize_data(
    wf_dict: dict, prefix: str | None = None, use_uuid: bool = False
) -> tuple[dict, dict, list]:
    """
    Serialize a nested workflow dictionary into a knowledge graph

    Args:
        wf_dict (dict): workflow dictionary
        prefix (str): prefix to be used for the nodes

    Returns:
        (tuple[dict, dict, list]): node_dict, channel_dict, edge_list
    """
    edge_list = []
    channel_dict = {}
    if prefix is None:
        prefix = wf_dict["label"]
        if use_uuid:
            prefix = f"{prefix}-{uuid.uuid4()}"
    node_dict = {
        prefix: {
            key: value
            for key, value in wf_dict.items()
            if key not in ["nodes", "edges"]
        }
    }
    node_dict[prefix]["function"] = get_function_metadata(wf_dict["function"])
    node_dict[prefix]["function"]["identifier"] = ".".join(
        (
            node_dict[prefix]["function"]["module"],
            node_dict[prefix]["function"]["qualname"],
            node_dict[prefix]["function"]["version"],
        )
    )
    for io_ in ["inputs", "outputs"]:
        for key, channel in wf_dict[io_].items():
            channel_label = _remove_us(prefix, io_, key)
            assert NS.PREFIX not in channel, f"{NS.PREFIX} already set"
            assert NS.TYPE not in channel, f"{NS.TYPE} already set"
            channel_dict[channel_label] = channel | {
                NS.PREFIX: prefix,
                NS.TYPE: io_,
            }
    for key, node in wf_dict.get("nodes", {}).items():
        child_node, child_channel, child_edges = serialize_data(
            node, prefix=_dot(prefix, key)
        )
        node_dict.update(child_node)
        edge_list.extend(child_edges)
        channel_dict.update(child_channel)
    for args in wf_dict.get("edges", []):
        edge_list.append([_remove_us(prefix, a) for a in args])
    return node_dict, channel_dict, edge_list


def _wf_data_to_networkx(node_dict, channel_dict, edge_list):
    G = nx.DiGraph()

    for key, data in channel_dict.items():
        G.add_node(
            key,
            step=data["semantikon_type"],
            **{k: v for k, v in data.items() if k not in ["semantikon_type"]},
        )

    for key, data in node_dict.items():
        if "." not in key:
            G.name = key
            continue
        G.add_node(
            key,
            step="node",
            **{k: v for k, v in data.items() if k not in ["inputs", "outputs"]},
        )
        for inp in data.get("inputs", {}).keys():
            G.add_edge(f"{key}.inputs.{inp}", key)
        for out in data.get("outputs", {}).keys():
            G.add_edge(key, f"{key}.outputs.{out}")

    for edge in edge_list:
        G.add_edge(*edge)
    mapping = {n: n.replace(".", "-") for n in G.nodes()}
    return nx.relabel_nodes(G, mapping, copy=True)


def _bundle_restrictions(g: Graph, only_dangling=True) -> list[BNode]:
    """
    Extract all OWL restriction BNodes from a graph.

    Args:
        g (Graph): RDF graph to search for restrictions

    Returns:
        (list[BNode]): list of BNodes that are OWL restrictions
    """
    return [
        r
        for r in g.subjects(RDF.type, OWL.Restriction)
        if len(list(g.subjects(None, r))) == 0 or not only_dangling
    ]


def _to_owl_restriction(
    pred: URIRef,
    target_class: URIRef,
    restriction_type: URIRef = OWL.someValuesFrom,
) -> Graph:
    g = Graph()
    restriction_node = BNode()

    # Build the restriction
    g.add((restriction_node, RDF.type, OWL.Restriction))
    g.add((restriction_node, OWL.onProperty, pred))
    g.add((restriction_node, restriction_type, target_class))

    return g


def _to_intersection(source_class: URIRef, list_items: list[URIRef]) -> Graph:
    g = Graph()
    intersection_node = BNode()
    list_head = BNode()
    g.add((intersection_node, RDF.type, OWL.Class))
    Collection(g, list_head, list_items)
    g.add((intersection_node, OWL.intersectionOf, list_head))

    g.add((source_class, RDFS.subClassOf, intersection_node))

    return g
