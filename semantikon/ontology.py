import copy
import json
from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from typing import TypeAlias, cast

import networkx as nx
from flowrep.tools import get_function_metadata
from owlrl import DeductiveClosure, RDFS_Semantics
from pyshacl import validate
from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import SH
from rdflib.term import IdentifiedNode

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
    denotes: URIRef = IAO["0000219"]
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
    if run_reasoner:
        DeductiveClosure(RDFS_Semantics).expand(g)
        _inherit_properties(g)
    return validate(g, shacl_graph=shacl)


def extract_dataclass(
    graph: Graph, ontology=SNS
) -> Graph:
    return Graph()


def get_knowledge_graph(
    wf_dict: dict,
    include_t_box: bool = True,
    include_a_box: bool = True,
) -> Graph:
    """
    Generate RDF graph from a dictionary containing workflow information

    Args:
        wf_dict (dict): dictionary containing workflow information
        include_t_box (bool): if True, include T-Box information
        include_a_box (bool): if True, include A-Box information

    Returns:
        (rdflib.Graph): graph containing workflow information
    """
    G = serialize_and_convert_to_networkx(wf_dict)
    graph = Graph()
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
    if include_t_box:
        graph += _nx_to_kg(G, t_box=True)
    if include_a_box:
        graph += _nx_to_kg(G, t_box=False)
    return graph


def _function_to_graph(
    f_node: URIRef,
    data: dict,
    input_args: list[dict],
    output_args: list[dict],
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
    g = Graph()
    g.add((f_node, RDF.type, SNS.software_method))
    g.add((f_node, RDFS.label, Literal(data["qualname"])))
    if data.get("docstring", "") != "":
        docstring = BNode(f_node + "_docstring")
        g.add((docstring, RDF.type, SNS.textual_entity))
        g.add((docstring, RDF.value, Literal(data["docstring"])))
        g.add((docstring, SNS.denotes, f_node))
    if uri is not None:
        g.add((f_node, SNS.is_about, uri))
    for io, io_args in zip(["input", "output"], [input_args, output_args]):
        for arg in io_args:
            arg_node = BNode("_".join([f_node, io, arg["arg"]]))
            if io == "input":
                g.add((arg_node, RDF.type, SNS.input_specification))
            else:
                g.add((arg_node, RDF.type, SNS.output_specification))
            g.add((arg_node, RDFS.label, Literal(arg["arg"])))
            g.add((f_node, SNS.has_parameter_specification, arg_node))
            g.add(
                (arg_node, SNS.has_parameter_position, Literal(arg.get("position", 0)))
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
    kg_node: URIRef,
    data: dict,
    G: SemantikonDiGraph,
    t_box: bool,
) -> Graph:
    g = Graph()
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
                    kg_node,
                    SNS.has_part,
                    BASE[item],
                )
        g.add((kg_node, RDFS.subClassOf, SNS.process))
        if "function" in data:
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
            g.add((kg_node, SNS.has_part, BNode(BASE[inp])))
        for out in G.successors(node_name):
            g.add((kg_node, SNS.has_part, BNode(BASE[out])))
        if "function" in data:
            g.add((kg_node, SNS.has_participant, f_node))
    return g


def _input_is_connected(io: str, G: SemantikonDiGraph) -> bool:
    candidate = list(G.predecessors(io))
    if len(candidate) == 1:
        if G.nodes[candidate[0]]["step"] == "node":
            return True
        return _input_is_connected(candidate[0], G)
    assert len(candidate) == 0
    return False


def _get_data_node(io: str, G: SemantikonDiGraph) -> BNode:
    candidate = list(G.predecessors(io))
    assert len(candidate) <= 1
    if len(candidate) == 0 or G.nodes[candidate[0]]["step"] == "node":
        return f"{io}_data"
    return _get_data_node(candidate[0], G)


def _detect_io_from_str(G: SemantikonDiGraph, seeked_io: str, ref_io: str) -> str:
    assert seeked_io.startswith("inputs") or seeked_io.startswith("outputs")
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
            result = BASE[_detect_io_from_str(G=G, seeked_io=t, ref_io=node_name)]
            if t_box:
                return result
            else:
                return BNode(result)

    g = Graph()
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
    restrictions: _rest_type, data_node: URIRef, predicate=RDFS.subClassOf
) -> Graph:
    g = Graph()
    assert isinstance(restrictions, tuple | list)
    assert isinstance(restrictions[0], tuple | list)
    if not isinstance(restrictions[0][0], tuple | list):
        restrictions = cast(_rest_type, (restrictions,))
    for r_set in restrictions:
        b_node = BNode("rest_" + sha256(str(r_set).encode("utf-8")).hexdigest())
        g.add((data_node, predicate, b_node))
        g.add((b_node, RDF.type, OWL.Restriction))
        for r in r_set:
            g.add((b_node, r[0], r[1]))
    return g


def _wf_io_to_graph(
    step: str,
    node_name: str,
    node: URIRef,
    data: dict,
    G: SemantikonDiGraph,
    t_box: bool,
) -> Graph:
    if not t_box:
        node = BNode(node)
    g = Graph()
    if data.get("label") is not None:
        g.add((node, RDFS.label, Literal(data["label"])))
    else:
        g.add((node, RDFS.label, Literal(node_name.split("-")[-1])))
    io_assignment = SNS.input_assignment if step == "inputs" else SNS.output_assignment
    data_node = BASE[_get_data_node(io=node_name, G=G)]
    has_specified_io = (
        SNS.has_specified_input if step == "inputs" else SNS.has_specified_output
    )
    if t_box:
        g += _to_owl_restriction(node, has_specified_io, data_node)
        g.add((node, RDFS.subClassOf, io_assignment))
        if step == "inputs" and _input_is_connected(node_name, G):
            out = list(G.predecessors(node_name))
            assert len(out) <= 1
            if len(out) == 1:
                assert G.nodes[out[0]]["step"] in ["outputs", "inputs"]
                if G.nodes[out[0]]["step"] == "outputs":
                    g += _to_owl_restriction(
                        BASE[out[0]], SNS.has_specified_output, data_node
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
                )
        if "restrictions" in data:
            assert step == "inputs", "restrictions only valid for inputs"
            g += _restrictions_to_triples(data["restrictions"], data_node=data_node)
        g.add((data_node, RDFS.subClassOf, SNS.value_specification))
    else:
        g.add((BNode(data_node), RDF.type, data_node))
        data_node = BNode(data_node)
        g.add((node, has_specified_io, data_node))
        if "value" in data and list(g.objects(data_node, RDF.value)) == []:
            g.add((data_node, RDF.value, Literal(data["value"])))
        if "units" in data and step == "outputs":
            g.add((data_node, QUDT.hasUnit, _units_to_uri(data["units"])))
        if "uri" in data and step == "outputs":
            bnode = BNode()
            g.add((bnode, RDF.type, data["uri"]))
            g.add((data_node, SNS.specifies_value_of, bnode))
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
        )
    return g


def _parse_precedes(
    G: SemantikonDiGraph,
    workflow_node: URIRef | BNode,
    t_box: bool,
) -> Graph:
    g = Graph()
    for node in G.nodes.data():
        if node[1]["step"] == "node":
            successors = list(_get_successor_nodes(G, node[0]))
            if len(successors) == 0:
                if t_box:
                    g += _to_owl_restriction(
                        workflow_node, SNS.has_part, BASE[node[0]]
                    )
                else:
                    g.add((workflow_node, SNS.has_part, BNode(BASE[node[0]])))
            else:
                if t_box:
                    for succ in successors:
                        g += _to_owl_restriction(
                            BASE[node[0]],
                            SNS.precedes,
                            BASE[succ],
                        )
                    g += _to_owl_restriction(
                        workflow_node,
                        SNS.has_part,
                        BASE[node[0]],
                    )
                else:
                    for succ in successors:
                        g.add(
                            (
                                BNode(BASE[node[0]]),
                                SNS.precedes,
                                BNode(BASE[succ]),
                            )
                        )
                    g.add((workflow_node, SNS.has_part, BNode(BASE[node[0]])))
    return g


def _parse_global_io(
    G: SemantikonDiGraph,
    workflow_node: URIRef | BNode,
    t_box: bool,
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
                    BASE[io[0]],
                )
            else:
                g.add((workflow_node, SNS.has_part, BNode(BASE[io[0]])))
    return g


def _nx_to_kg(G: SemantikonDiGraph, t_box: bool) -> Graph:
    g = Graph()
    workflow_node = BASE[G.name] if t_box else BNode(BASE[G.name])
    for comp in G.nodes.data():
        data = comp[1].copy()
        step = data.pop("step")
        node = BASE[comp[0]]
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
            )
        else:
            g += _wf_io_to_graph(
                step=step,
                node_name=comp[0],
                node=node,
                data=data,
                G=G,
                t_box=t_box,
            )

    if t_box:
        g.add((workflow_node, RDF.type, OWL.Class))
        g.add((workflow_node, RDFS.subClassOf, SNS.process))
    else:
        g.add((workflow_node, RDF.type, BASE[G.name]))
    g += _parse_precedes(G=G, workflow_node=workflow_node, t_box=t_box)
    g += _parse_global_io(G=G, workflow_node=workflow_node, t_box=t_box)
    return g


def _get_successor_nodes(G, node_name):
    for out in G.successors(node_name):
        for inp in G.successors(out):
            for node in G.successors(inp):
                yield node


def serialize_and_convert_to_networkx(
    wf_dict: dict, prefix: str | None = None
) -> SemantikonDiGraph:
    """
    Serializes a workflow dictionary and converts it into a NetworkX directed graph.

    Args:
        wf_dict (dict): The workflow dictionary to process.
        prefix (str | None): Optional prefix for node names.

    Returns:
        SemantikonDiGraph: A directed graph representation of the workflow.
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
        assert "function" in wf_dict or wf_dict["type"] != "Function"
        if "function" in wf_dict:
            node_dict[prefix]["function"] = get_function_metadata(
                wf_dict["function"], full_metadata=True
            )
            node_dict[prefix]["function"]["identifier"] = ".".join(
                (
                    node_dict[prefix]["function"]["module"],
                    node_dict[prefix]["function"]["qualname"],
                    node_dict[prefix]["function"]["version"],
                )
            )
        for io_ in ["inputs", "outputs"]:
            for loc, (key, channel) in enumerate(wf_dict[io_].items()):
                channel_label = _remove_us(prefix, io_, key)
                assert "semantikon_type" not in channel, "semantikon_type already set"
                channel_dict[channel_label] = channel | {
                    "semantikon_type": io_,
                    "position": channel.get("position", loc),
                    "arg": key,
                }
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
    ) -> SemantikonDiGraph:
        """Build a NetworkX directed graph from node, channel, and edge data."""
        G = SemantikonDiGraph()

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


def owl_restrictions_to_shacl(
    owl_graph: Graph, excluded_nodes: list[BNode] | None = None
) -> Graph:
    def iter_supported_restrictions(g: Graph):
        """
        Yield (base_class, restriction_node, property, restriction_type, value)
        for supported OWL restrictions.
        """
        for r in g.subjects(RDF.type, OWL.Restriction):
            if excluded_nodes is not None and r in excluded_nodes:
                continue
            prop = g.value(r, OWL.onProperty)
            if prop is None:
                continue
            for restriction_type in (OWL.someValuesFrom, OWL.hasValue):
                value = g.value(r, restriction_type)
                if value is None:
                    continue

                for base_cls in g.subjects(RDFS.subClassOf, r):
                    yield base_cls, prop, restriction_type, value

    shacl_graph = Graph()
    node_shapes = {}

    for base_cls, prop, rtype, value in iter_supported_restrictions(owl_graph):

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
