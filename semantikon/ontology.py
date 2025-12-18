from dataclasses import dataclass
from typing import Any, TypeAlias

import networkx as nx
from flowrep.tools import get_function_metadata
from owlrl import DeductiveClosure, OWLRL_Semantics
from rdflib import OWL, PROV, RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import SH
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
    software_method: URIRef = IAO["0000591"]


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
    namespace: Namespace | None = None,
) -> Graph:
    """
    Generate RDF graph from a dictionary containing workflow information

    Args:
        wf_dict (dict): dictionary containing workflow information
        t_box (bool): if True, generate T-Box graph, otherwise A-Box graph
        namespace (Namespace): namespace to be used for the graph

    Returns:
        (rdflib.Graph): graph containing workflow information
    """
    node_dict, channel_dict, edge_list = serialize_data(wf_dict)
    G = _wf_data_to_networkx(node_dict, channel_dict, edge_list)
    graph = _nx_to_kg(G, t_box=t_box, namespace=namespace)
    return graph


def _wf_node_to_graph(
    node_name: str,
    kg_node: URIRef,
    data: dict,
    G: nx.DiGraph,
    t_box: bool,
    namespace: Namespace,
) -> Graph:
    g = Graph()
    f_node = namespace[data["function"]["identifier"].replace(".", "-")]
    g.add((f_node, RDF.type, SNS.software_method))
    if t_box:
        for io in [G.predecessors(node_name), G.successors(node_name)]:
            for item in io:
                g += _to_owl_restriction(
                    on_property=SNS.has_part,
                    target_class=namespace[item],
                    base_node=kg_node,
                )
        g.add((kg_node, RDFS.subClassOf, SNS.process))
        g += _to_owl_restriction(
            on_property=SNS.has_participant,
            target_class=f_node,
            base_node=kg_node,
        )
    else:
        g.add((kg_node, RDF.type, SNS.process))
        for inp in G.predecessors(node_name):
            g.add((kg_node, SNS.has_part, namespace[inp]))
        for out in G.successors(node_name):
            g.add((kg_node, SNS.has_part, namespace[out]))
        g.add((kg_node, SNS.has_participant, f_node))
    return g


def _get_data_node(io: str, G: nx.DiGraph, t_box: bool = False) -> BNode:
    candidate = list(G.predecessors(io))
    assert len(candidate) <= 1
    if len(candidate) == 0 or G.nodes[candidate[0]]["step"] == "node" or t_box:
        return f"{io}_data"
    return _get_data_node(candidate[0], G)


def _wf_io_to_graph(
    step: str,
    node_name: str,
    node: URIRef,
    data: dict,
    G: nx.DiGraph,
    t_box: bool,
    namespace: Namespace,
) -> Graph:
    g = Graph()
    if data.get("label") is not None:
        g.add(node, RDFS.label, Literal(data["label"]))
    io_assignment = SNS.input_assignment if step == "inputs" else SNS.output_assignment
    data_node = BNode(namespace[_get_data_node(io=node_name, G=G, t_box=t_box)])
    has_specified_io = (
        SNS.has_specified_input if step == "inputs" else SNS.has_specified_output
    )
    if t_box:
        g += _to_owl_restriction(has_specified_io, data_node, base_node=node)
        g.add((node, RDFS.subClassOf, io_assignment))
        if "units" in data:
            g += _to_owl_restriction(
                on_property=SNS.has_unit,
                target_class=_units_to_uri(data["units"]),
                restriction_type=OWL.hasValue,
                base_node=data_node,
            )
        if step == "inputs":
            out = list(G.predecessors(node_name))
            assert len(out) <= 1
            if len(out) == 1:
                assert G.nodes[out[0]]["step"] in ["outputs", "inputs"]
                if G.nodes[out[0]]["step"] == "outputs":
                    g += _to_owl_restriction(
                        on_property=SNS.is_specified_output_of,
                        target_class=namespace[out[0]],
                        base_node=data_node,
                    )
        g.add((data_node, RDFS.subClassOf, SNS.value_specification))
        if "uri" in data:
            g += _to_owl_restriction(
                on_property=SNS.specifies_value_of,
                target_class=data["uri"],
                base_node=data_node,
            )
    else:
        g.add((node, RDF.type, io_assignment))
        g.add((node, has_specified_io, data_node))
        if "value" in data and list(g.objects(data_node, RDF.value)) == []:
            g.add((data_node, RDF.value, Literal(data["value"])))
        if "units" in data:
            g.add((data_node, OWL.hasValue, _units_to_uri(data["units"])))
        if "uri" in data:
            g.add((data_node, RDF.type, data["uri"]))
        g.add((data_node, SNS.specifies_value_of, SNS.value_specification))
    if "derived_from" in data:
        assert step == "outputs", "derived_from only valid for outputs"
        for inp in G.predecessors(node_name.split("-outputs-")[0]):
            if inp.endswith(data["derived_from"].replace(".", "-")):
                if t_box:
                    g += _to_owl_restriction(
                        on_property=PROV.wasDerivedFrom,
                        target_class=BNode(namespace[f"{inp}_data"]),
                        base_node=data_node,
                    )
                else:
                    g.add(
                        (
                            data_node,
                            PROV.wasDerivedFrom,
                            BNode(namespace[f"{inp}_data"]),
                        )
                    )
                break
        else:
            raise ValueError(
                f"derived_from {data['derived_from']} not found in predecessors"
            )
    return g


def _parse_precedes(
    G: nx.DiGraph,
    workflow_node: URIRef,
    t_box: bool,
    namespace: Namespace,
) -> Graph:
    g = Graph()
    for node in G.nodes.data():
        if node[1]["step"] == "node":
            successors = list(_get_successor_nodes(G, node[0]))
            if len(successors) == 0:
                if t_box:
                    g += _to_owl_restriction(
                        on_property=SNS.has_part,
                        target_class=namespace[node[0]],
                        base_node=workflow_node,
                    )
                else:
                    g.add((workflow_node, SNS.has_part, namespace[node[0]]))
            else:
                if t_box:
                    node_tmp = BNode()
                    for succ in successors:
                        g += _to_owl_restriction(
                            on_property=SNS.precedes,
                            target_class=namespace[succ],
                            base_node=node_tmp,
                        )
                    g.add((node_tmp, RDFS.subClassOf, namespace[node[0]]))
                    g += _to_owl_restriction(
                        on_property=SNS.has_part,
                        target_class=node_tmp,
                        base_node=workflow_node,
                    )
                else:
                    for succ in successors:
                        g.add((namespace[node[0]], SNS.precedes, namespace[succ]))
                    g.add((workflow_node, SNS.has_part, namespace[node[0]]))
    return g


def _parse_global_io(
    G: nx.DiGraph,
    workflow_node: URIRef,
    t_box: bool,
    namespace: Namespace,
) -> Graph:
    g = Graph()
    global_inputs = [
        n for n in G.nodes.data() if G.in_degree(n[0]) == 0 and n[1]["step"] == "inputs"
    ]
    global_outputs = [
        n
        for n in G.nodes.data()
        if G.out_degree(n[0]) == 0 and n[1]["step"] == "outputs"
    ]
    for on_property, global_io in zip(
        [SNS.has_specified_input, SNS.has_specified_output],
        [global_inputs, global_outputs],
    ):
        for io in global_io:
            if t_box:
                g += _to_owl_restriction(
                    on_property=on_property,
                    target_class=namespace[io[0]],
                    base_node=workflow_node,
                )
            else:
                g.add((workflow_node, on_property, namespace[io[0]]))
    return g


def _nx_to_kg(G: nx.DiGraph, t_box: bool, namespace: Namespace | None = None) -> Graph:
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
    if namespace is not None:
        graph.bind("ns", str(namespace))
    if namespace is None:
        namespace = BASE
    workflow_node = namespace[G.name]
    for comp in G.nodes.data():
        data = comp[1].copy()
        step = data.pop("step")
        node = namespace[comp[0]]
        if t_box:
            g.add((node, RDF.type, OWL.Class))
        assert step in ["node", "inputs", "outputs"], f"Unknown step: {step}"
        if step == "node":
            g += _wf_node_to_graph(
                node_name=comp[0],
                kg_node=node,
                data=data,
                G=G,
                t_box=t_box,
                namespace=namespace,
            )
        else:
            g += _wf_io_to_graph(
                step=step,
                node_name=comp[0],
                node=node,
                data=data,
                G=G,
                t_box=t_box,
                namespace=namespace,
            )

    g.add((workflow_node, RDF.type, OWL.Class))
    g.add((workflow_node, RDFS.subClassOf, SNS.process))
    g += _parse_precedes(
        G=G, workflow_node=workflow_node, t_box=t_box, namespace=namespace
    )
    g += _parse_global_io(
        G=G, workflow_node=workflow_node, t_box=t_box, namespace=namespace
    )
    return g


def _get_successor_nodes(G, node_name):
    for out in G.successors(node_name):
        for inp in G.successors(out):
            for node in G.successors(inp):
                yield node


def serialize_data(wf_dict: dict, prefix: str | None = None) -> tuple[dict, dict, list]:
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
            assert "semantikon_type" not in channel, "semantikon_type already set"
            channel_dict[channel_label] = channel | {"semantikon_type": io_}
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


def _to_owl_restriction(
    on_property: URIRef,
    target_class: URIRef,
    restriction_node: BNode | URIRef | None = None,
    restriction_type: URIRef = OWL.someValuesFrom,
    base_node: URIRef | None = None,
) -> Graph:
    g = Graph()
    if restriction_node is None:
        restriction_node = BNode()

    # Build the restriction
    g.add((restriction_node, RDF.type, OWL.Restriction))
    g.add((restriction_node, OWL.onProperty, on_property))
    g.add((restriction_node, restriction_type, target_class))
    if base_node is not None:
        g.add((base_node, RDFS.subClassOf, restriction_node))

    return g


def _get_graph_hash(G: nx.DiGraph) -> str:
    """
    Generate a hash for a NetworkX graph, making sure that data types and
    values (except for the global ones) because they can often not be
    serialized.

    Args:
        G (nx.DiGraph): input graph

    Returns:
        (str): hash of the graph
    """
    G_tmp = G.copy()
    for node in G_tmp.nodes:
        if G_tmp.in_degree(node) > 0:
            if "value" in G_tmp.nodes[node]:
                del G_tmp.nodes[node]["value"]
        if "dtype" in G_tmp.nodes[node]:
            del G_tmp.nodes[node]["dtype"]
    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G_tmp)

def _to_shacl_shape(
    on_property: URIRef,
    target_class: URIRef,
    shape_node: BNode | URIRef | None = None,
    base_node: URIRef | None = None,
    min_count: int | None = 1,
) -> Graph:
    g = Graph()

    if shape_node is None:
        shape_node = BNode()

    # Declare PropertyShape
    g.add((shape_node, RDF.type, SH.PropertyShape))
    g.add((shape_node, SH.path, on_property))
    g.add((shape_node, SH["class"], target_class))

    # Existential semantics (OWL someValuesFrom analogue)
    if min_count is not None:
        g.add((shape_node, SH.minCount, Literal(min_count)))

    # Attach to a NodeShape via targetClass
    if base_node is not None:
        node_shape = BNode()
        g.add((node_shape, RDF.type, SH.NodeShape))
        g.add((node_shape, SH.targetClass, base_node))
        g.add((node_shape, SH.property, shape_node))

    return g
