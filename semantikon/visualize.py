import networkx as nx
from graphviz import Digraph
from hashlib import sha256
from rdflib import Graph, RDFS, URIRef


def _get_triples(graph: Graph):
    rest_types = {
        "tapered": "owl:someValuesFrom",
        "dashed": "owl:allValuesFrom",
        "bold": "owl:hasValue",
    }
    query = """
    SELECT ?parent ?property ?child WHERE {
        ?parent rdfs:subClassOf ?bnode .
        ?bnode a owl:Restriction .
        ?bnode owl:onProperty ?property .
        ?bnode R_TYPE ?child .
    }"""
    for style, rest in rest_types.items():
        for subj, pred, obj in graph.query(query.replace("R_TYPE", rest)):
            yield subj, pred, obj, style


def _rename_predicate(pred: str) -> str:
    edge_dict = {
        "bfo:0000051": "bfo:has_part",
        "bfo:0000063": "bfo:precedes",
        "iao:0000235": "iao:denoted_by",
        "obi:0000293": "obi:has_specified_input",
        "obi:0000299": "obi:has_specified_output",
        "obi:0001927": "obi:specified_values_of",
        "ro:0000057": "pmd:output_assignment",
    }
    return edge_dict.get(*2 * [pred])


def _get_node_color(comp: str, graph: Graph) -> str:
    color_dict = {
        "bfo:0000015": "red",
        "obi:0001933": "yellow",
        "pmdco:0000066": "green",
        "pmdco:0000067": "blue",
    }
    parent_classes = [
        item
        for item in graph.objects(comp, RDFS.subClassOf)
        if isinstance(item, URIRef)
    ]
    for cl in parent_classes:
        if graph.qname(cl) in color_dict:
            return color_dict[graph.qname(cl)]
    return "black"


def _rdflib_to_nx(graph: Graph) -> nx.DiGraph:
    G = nx.DiGraph()
    for subj, pred, obj, style in _get_triples(graph):
        for part in [subj, obj]:
            if part in G.nodes:
                continue
            G.add_node(graph.qname(part), color=_get_node_color(part, graph))
        G.add_edge(
            graph.qname(subj),
            graph.qname(obj),
            label=_rename_predicate(graph.qname(pred)),
            style=style,
        )
    return G


def visualize_recipe(graph: Graph) -> Digraph:
    G = _rdflib_to_nx(graph)
    dot = Digraph()
    for node, data in G.nodes.data():
        dot.node(sha256(node.encode()).hexdigest(), node, **data)
    for subj, obj, data in G.edges.data():
        dot.edge(
            sha256(subj.encode()).hexdigest(), sha256(obj.encode()).hexdigest(), **data
        )
    return dot
