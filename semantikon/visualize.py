from hashlib import sha256

import networkx as nx
from graphviz import Digraph
from rdflib import OWL, RDF, RDFS, Graph, URIRef


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
        "obi:0001927": "obi:specifies_values_of",
        "ro:0000057": "pmd:output_assignment",
    }
    return edge_dict.get(*2 * [pred])


def _color_predicate(pred: str) -> str:
    edge_dict = {
        "bfo:has_part": "darkblue",
        "bfo:precedes": "brown",
        "iao:denoted_by": "darkgreen",
        "obi:has_specified_input": "darkorange",
        "obi:has_specified_output": "darkcyan",
        "obi:specifies_values_of": "darkviolet",
        "pmd:output_assignment": "forestgreen",
    }
    return edge_dict.get(pred, "black")


def _get_node_color(comp: str, graph: Graph) -> str:
    subclass_dict = {
        "bfo:0000015": "lightpink",
        "obi:0001933": "lightyellow",
        "pmdco:0000066": "lightgreen",
        "pmdco:0000067": "lightblue",
    }
    type_dict = {"iao:0000591": "lightsalmon"}
    for pred, d in zip([RDFS.subClassOf, RDF.type], [subclass_dict, type_dict]):
        parent_classes = [
            item for item in graph.objects(comp, pred) if isinstance(item, URIRef)
        ]
        for cl in parent_classes:
            if graph.qname(cl) in d:
                return d[graph.qname(cl)]
    return "white"


def _is_class(term: URIRef, graph: Graph) -> bool:
    if (term, RDF.type, OWL.Class) in graph:
        return True
    if len(list(graph.objects(term, RDF.type))) > 0:
        return False
    if len(list(graph.subjects(OWL.hasValue, term))) > 0:
        return False
    return True


def _rdflib_to_nx(graph: Graph) -> nx.DiGraph:
    G = nx.DiGraph()
    for subj, pred, obj, style in _get_triples(graph):
        for part in [subj, obj]:
            if part in G.nodes:
                continue
            G.add_node(
                graph.qname(part),
                fillcolor=_get_node_color(part, graph),
                style="filled" if _is_class(part, graph) else "filled,rounded,dashed",
                shape="box",
            )
        label = _rename_predicate(graph.qname(pred))
        color = _color_predicate(label)
        G.add_edge(
            graph.qname(subj),
            graph.qname(obj),
            label=label,
            style=style,
            color=color,
            fontcolor=color,
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
