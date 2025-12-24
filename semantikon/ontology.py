import json
from dataclasses import dataclass
from typing import Any, TypeAlias

import networkx as nx
from flowrep.tools import get_function_metadata
from owlrl import DeductiveClosure, OWLRL_Semantics
from pyshacl import validate
from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
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


def validate_values(wf_dict: dict) -> tuple:
    """
    Validate if all values required by restrictions are present in the graph

    Args:
        wf_dict (dict): dictionary containing workflow information

    Returns:
        (tuple): list of missing triples
    """
    g_t = get_knowledge_graph(wf_dict, t_box=True)
    g_a = get_knowledge_graph(wf_dict, t_box=False)
    shacl = owl_restrictions_to_shacl(g_t)
    return validate(g_a, shacl_graph=shacl)


def extract_dataclass(
    graph: Graph, namespace: Namespace | None = None, ontology=SNS
) -> Graph:
    return Graph()


def get_knowledge_graph(
    wf_dict: dict,
    t_box: bool = False,
) -> Graph:
    """
    Generate RDF graph from a dictionary containing workflow information

    Args:
        wf_dict (dict): dictionary containing workflow information
        t_box (bool): if True, generate T-Box graph, otherwise A-Box graph

    Returns:
        (rdflib.Graph): graph containing workflow information
    """
    G = serialize_and_convert_to_networkx(wf_dict)
    graph = _nx_to_kg(G, t_box=t_box)
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
                    kg_node,
                    SNS.has_part,
                    namespace[item],
                )
        g.add((kg_node, RDFS.subClassOf, SNS.process))
        g += _to_owl_restriction(
            kg_node,
            SNS.has_participant,
            f_node,
            restriction_type=OWL.hasValue,
        )
    else:
        g.add((BNode(kg_node), RDF.type, kg_node))
        kg_node = BNode(kg_node)
        for inp in G.predecessors(node_name):
            g.add((kg_node, SNS.has_part, BNode(namespace[inp])))
        for out in G.successors(node_name):
            g.add((kg_node, SNS.has_part, BNode(namespace[out])))
        g.add((kg_node, SNS.has_participant, f_node))
    return g


def _get_data_node(io: str, G: nx.DiGraph) -> BNode:
    candidate = list(G.predecessors(io))
    assert len(candidate) <= 1
    if len(candidate) == 0 or G.nodes[candidate[0]]["step"] == "node":
        return f"{io}_data"
    return _get_data_node(candidate[0], G)


def _detect_io_from_str(G: nx.DiGraph, seeked_io: str, ref_io: str) -> str:
    main_node = ref_io.replace(".", "-").split("-outputs-")[0].split("-inputs-")[0]
    candidate = (
        G.predecessors(main_node) if "inputs" in seeked_io else G.successors(main_node)
    )
    for io in candidate:
        if io.endswith(seeked_io.replace(".", "-")):
            return _get_data_node(io=io, G=G)
    raise ValueError(f"IO {seeked_io} not found in graph")


def _translate_triples(
    triples: _triple_type,
    node_name: str,
    data_node: BNode,
    G: nx.DiGraph,
    t_box: bool,
    namespace: Namespace,
) -> Graph:
    def _local_str_to_uriref(t: URIRef | BNode | str | None) -> IdentifiedNode:
        if isinstance(t, (URIRef, BNode)):
            return t
        elif t == "self" or t is None:
            return data_node
        elif isinstance(t, str) and (t.startswith("inputs") or t.startswith("outputs")):
            result = namespace[_detect_io_from_str(G=G, seeked_io=t, ref_io=node_name)]
            if t_box:
                return result
            else:
                return BNode(result)
        else:
            raise ValueError(f"{t} not recognized")

    g = Graph()
    for triple in triples:
        if len(triple) == 2:
            s = data_node
            p, o = triple
        else:
            s, p, o = triple
        s = _local_str_to_uriref(s)
        o = _local_str_to_uriref(o)
        if t_box:
            g += _to_owl_restriction(s, p, o)
        else:
            g.add((s, p, o))
    return g


def _wf_io_to_graph(
    step: str,
    node_name: str,
    node: URIRef,
    data: dict,
    G: nx.DiGraph,
    t_box: bool,
    namespace: Namespace,
) -> Graph:
    if not t_box:
        node = BNode(node)
    g = Graph()
    if data.get("label") is not None:
        g.add((node, RDFS.label, Literal(data["label"])))
    else:
        g.add((node, RDFS.label, Literal(node_name.split("-")[-1])))
    io_assignment = SNS.input_assignment if step == "inputs" else SNS.output_assignment
    data_node = namespace[_get_data_node(io=node_name, G=G)]
    has_specified_io = (
        SNS.has_specified_input if step == "inputs" else SNS.has_specified_output
    )
    if t_box:
        g += _to_owl_restriction(node, has_specified_io, data_node)
        g.add((node, RDFS.subClassOf, io_assignment))
        if "units" in data:
            g += _to_owl_restriction(
                base_node=data_node,
                on_property=QUDT.hasUnit,
                target_class=_units_to_uri(data["units"]),
                restriction_type=OWL.hasValue,
            )
        if step == "inputs":
            out = list(G.predecessors(node_name))
            assert len(out) <= 1
            if len(out) == 1:
                assert G.nodes[out[0]]["step"] in ["outputs", "inputs"]
                if G.nodes[out[0]]["step"] == "outputs":
                    g += _to_owl_restriction(
                        namespace[out[0]], SNS.has_specified_output, data_node
                    )
        g.add((data_node, RDFS.subClassOf, SNS.value_specification))
        if "uri" in data:
            g += _to_owl_restriction(
                data_node,
                SNS.specifies_value_of,
                data["uri"],
                restriction_type=OWL.hasValue,
            )
    else:
        g.add((BNode(data_node), RDF.type, data_node))
        data_node = BNode(data_node)
        g.add((node, has_specified_io, data_node))
        if "value" in data and list(g.objects(data_node, RDF.value)) == []:
            g.add((data_node, RDF.value, Literal(data["value"])))
        if "units" in data:
            g.add((data_node, QUDT.hasUnit, _units_to_uri(data["units"])))
        if "uri" in data:
            g.add((data_node, SNS.specifies_value_of, data["uri"]))
    triples = data.get("triples", [])
    if triples != [] and not isinstance(triples[0], list | tuple):
        triples = [triples]
    if "derived_from" in data:
        assert step == "outputs", "derived_from only valid for outputs"
        triples.append(("self", SNS.derives_from, data["derived_from"]))
    if len(triples) > 0:
        g += _translate_triples(
            triples=triples,
            node_name=node_name,
            data_node=data_node,
            G=G,
            t_box=t_box,
            namespace=namespace,
        )
    return g


def _parse_precedes(
    G: nx.DiGraph,
    workflow_node: URIRef | BNode,
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
                        workflow_node, SNS.has_part, namespace[node[0]]
                    )
                else:
                    g.add((workflow_node, SNS.has_part, BNode(namespace[node[0]])))
            else:
                if t_box:
                    for succ in successors:
                        g += _to_owl_restriction(
                            namespace[node[0]],
                            SNS.precedes,
                            namespace[succ],
                        )
                    g += _to_owl_restriction(
                        workflow_node,
                        SNS.has_part,
                        namespace[node[0]],
                    )
                else:
                    for succ in successors:
                        g.add(
                            (
                                BNode(namespace[node[0]]),
                                SNS.precedes,
                                BNode(namespace[succ]),
                            )
                        )
                    g.add((workflow_node, SNS.has_part, BNode(namespace[node[0]])))
    return g


def _parse_global_io(
    G: nx.DiGraph,
    workflow_node: URIRef | BNode,
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
    for global_io in [global_inputs, global_outputs]:
        for io in global_io:
            if t_box:
                g += _to_owl_restriction(
                    workflow_node,
                    SNS.has_part,
                    namespace[io[0]],
                )
            else:
                g.add((workflow_node, SNS.has_part, BNode(namespace[io[0]])))
    return g


def _nx_to_kg(G: nx.DiGraph, t_box: bool) -> Graph:
    g = Graph()
    g.bind("qudt", str(QUDT))
    g.bind("unit", "http://qudt.org/vocab/unit/")
    g.bind("sns", str(BASE))
    g.bind("iao", str(IAO))
    g.bind("bfo", str(BFO))
    g.bind("obi", str(OBI))
    g.bind("ro", str(RO))
    g.bind("pmdco", str(PMD))
    g.bind("schema", str(SCHEMA))
    g.bind("stato", str(STATO))
    namespace = BASE
    workflow_node = namespace[G.name] if t_box else BNode(namespace[G.name])
    for comp in G.nodes.data():
        data = comp[1].copy()
        step = data.pop("step")
        node = namespace[comp[0]]
        if t_box:
            g.add((node, RDF.type, OWL.Class))
        else:
            g.add((BNode(node), RDF.type, BASE[comp[0]]))
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

    if t_box:
        g.add((workflow_node, RDF.type, OWL.Class))
        g.add((workflow_node, RDFS.subClassOf, SNS.process))
    else:
        g.add((workflow_node, RDF.type, BASE[G.name]))
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


def serialize_and_convert_to_networkx(
    wf_dict: dict, prefix: str | None = None
) -> nx.DiGraph:
    """
    Serializes a workflow dictionary and converts it into a NetworkX directed graph.

    Args:
        wf_dict (dict): The workflow dictionary to process.
        prefix (str | None): Optional prefix for node names.

    Returns:
        nx.DiGraph: A directed graph representation of the workflow.
    """

    def _remove_us(*args) -> str:
        """Remove underscores from the components of a dotted string."""
        s = ".".join(args)
        return ".".join(part.split("__")[-1] for part in s.split("."))

    def _dot(*args) -> str:
        """Join components with a dot, ignoring None values."""
        return ".".join([a for a in args if a is not None])

    def _serialize_workflow(wf_dict: dict, prefix: str) -> tuple[dict, dict, list]:
        """Serialize the workflow dictionary into node, channel, and edge data."""
        edge_list = []
        channel_dict = {}
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
            child_node, child_channel, child_edges = _serialize_workflow(
                node, prefix=_dot(prefix, key)
            )
            node_dict.update(child_node)
            edge_list.extend(child_edges)
            channel_dict.update(child_channel)
        for args in wf_dict.get("edges", []):
            edge_list.append([_remove_us(prefix, a) for a in args])
        return node_dict, channel_dict, edge_list

    def _build_graph(
        node_dict: dict, channel_dict: dict, edge_list: list
    ) -> nx.DiGraph:
        """Build a NetworkX directed graph from node, channel, and edge data."""
        G = nx.DiGraph()

        # Add channel nodes
        for key, data in channel_dict.items():
            G.add_node(
                key,
                step=data["semantikon_type"],
                **{k: v for k, v in data.items() if k not in ["semantikon_type"]},
            )

        # Add workflow and node nodes
        for key, data in node_dict.items():
            if "." not in key:  # Root workflow node
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

        # Add edges
        for edge in edge_list:
            G.add_edge(*edge)

        # Relabel nodes to replace dots with dashes
        mapping = {n: n.replace(".", "-") for n in G.nodes()}
        return nx.relabel_nodes(G, mapping, copy=True)

    # Main logic
    if prefix is None:
        prefix = wf_dict["label"]
    node_dict, channel_dict, edge_list = _serialize_workflow(wf_dict, prefix)
    return _build_graph(node_dict, channel_dict, edge_list)


def _to_owl_restriction(
    base_node: URIRef | None,
    on_property: URIRef,
    target_class: URIRef,
    restriction_type: URIRef = OWL.someValuesFrom,
) -> Graph:
    g = Graph()
    restriction_node = BNode()

    # Build the restriction
    g.add((restriction_node, RDF.type, OWL.Restriction))
    g.add((restriction_node, OWL.onProperty, on_property))
    g.add((restriction_node, restriction_type, target_class))
    if base_node is not None:
        g.add((base_node, RDFS.subClassOf, restriction_node))

    return g


def _get_graph_hash(G: nx.DiGraph, with_global_inputs: bool = True) -> str:
    """
    Generate a hash for a NetworkX graph, making sure that data types and
    values (except for the global ones) because they can often not be
    serialized.

    Args:
        G (nx.DiGraph): input graph
        with_global_inputs (bool): if True, keep values for global inputs

    Returns:
        (str): hash of the graph
    """
    G_tmp = G.copy()
    for node in G_tmp.nodes:
        if G_tmp.in_degree(node) > 0 or not with_global_inputs:
            if "value" in G_tmp.nodes[node]:
                del G_tmp.nodes[node]["value"]
        if "dtype" in G_tmp.nodes[node]:
            del G_tmp.nodes[node]["dtype"]

    for _, attrs in G_tmp.nodes(data=True):
        attrs["canon"] = json.dumps(attrs, sort_keys=True)

    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
        G_tmp, node_attr="canon"
    )


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


def owl_restrictions_to_shacl(owl_graph: Graph) -> Graph:
    def iter_supported_restrictions(g: Graph):
        """
        Yield (base_class, restriction_node, property, restriction_type, value)
        for supported OWL restrictions.
        """
        for r in g.subjects(RDF.type, OWL.Restriction):
            prop = g.value(r, OWL.onProperty)
            if prop is None:
                continue

            for restriction_type in (OWL.someValuesFrom, OWL.hasValue):
                value = g.value(r, restriction_type)
                if value is None:
                    continue

                for base_cls in g.subjects(RDFS.subClassOf, r):
                    yield base_cls, r, prop, restriction_type, value

    shacl_graph = Graph()
    node_shapes = {}

    for base_cls, r, prop, rtype, value in iter_supported_restrictions(owl_graph):

        # One NodeShape per base class
        if base_cls not in node_shapes:
            ns = BNode()
            node_shapes[base_cls] = ns
            shacl_graph.add((ns, RDF.type, SH.NodeShape))
            shacl_graph.add((ns, SH.targetClass, base_cls))
        else:
            ns = node_shapes[base_cls]

        ps = BNode()
        shacl_graph.add((ps, RDF.type, SH.PropertyShape))
        shacl_graph.add((ps, SH.path, prop))

        if rtype == OWL.someValuesFrom:
            # Existential restriction:
            # ∃ prop . C  →  qualifiedValueShape + qualifiedMinCount
            qvs = BNode()
            shacl_graph.add((qvs, SH["class"], value))

            shacl_graph.add((ps, SH.qualifiedValueShape, qvs))
            shacl_graph.add((ps, SH.qualifiedMinCount, Literal(1)))

        elif rtype == OWL.hasValue:
            shacl_graph.add((ps, SH.hasValue, value))

        shacl_graph.add((ns, SH.property, ps))

    return shacl_graph
