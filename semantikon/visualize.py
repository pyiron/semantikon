from hashlib import sha256

import networkx as nx
from graphviz import Digraph
from rdflib import OWL, RDF, RDFS, Graph, URIRef

subclass_color_dict = {
    "pmdco:0000011": "lightpink",
    "obi:0001933": "lightyellow",
    "pmdco:0000066": "lightgreen",
    "pmdco:0000067": "lightblue",
}
type_color_dict = {"iao:0000591": "lightsalmon"}


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
        "obi:0001927": "obi:specifies_values_of",
        "ro:0000057": "ro:has_participant",
        "ro:0000059": "ro:concretizes",
    }
    return edge_dict.get(*2 * [pred])


def _color_predicate(pred: str) -> str:
    edge_dict = {
        "bfo:has_part": "darkblue",
        "bfo:precedes": "brown",
        "iao:denoted_by": "darkgreen",
        "obi:specifies_values_of": "darkviolet",
        "ro:has_participant": "darkorange",
        "ro:concretizes": "darkcyan",
    }
    return edge_dict.get(pred, "black")


def _get_parent_class(comp: str, graph: Graph) -> str:
    for pred in [RDFS.subClassOf, RDF.type]:
        parent_classes = [
            item for item in graph.objects(comp, pred) if isinstance(item, URIRef)
        ]
        for cl in parent_classes:
            if (
                graph.qname(cl) in subclass_color_dict
                or graph.qname(cl) in type_color_dict
            ):
                return graph.qname(cl)
    return ""


def _get_node_color(comp: str, graph: Graph) -> str:
    parent_class = _get_parent_class(comp, graph)
    if parent_class in subclass_color_dict:
        return subclass_color_dict[parent_class]
    if parent_class in type_color_dict:
        return type_color_dict[parent_class]
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
                parent_class=_get_parent_class(part, graph),
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


def _to_node(key: str, parent_class: str) -> str:
    translation = {
        "pmdco:0000011": "workflow_node",
        "obi:0001933": "value_specification",
        "pmdco:0000066": "input_assignment",
        "pmdco:0000067": "output_assignment",
        "iao:0000591": "software_method",
    }
    text = translation.get(*2 * [parent_class]) + " / " + parent_class
    rows = '<<table border="0" cellborder="0" cellspacing="0">'
    rows += f"<tr><td align='center'><U>{key}</U></td></tr>"
    if len(parent_class) > 0:
        rows += f"<tr><td><I>{text}</I></td></tr>"
    rows += "</table>>"
    return rows


def visualize_recipe(graph: Graph) -> Digraph:
    G = _rdflib_to_nx(graph)
    dot = Digraph()
    for node, data in G.nodes.data():
        cell = _to_node(node, data.pop("parent_class"))
        dot.node(sha256(node.encode()).hexdigest(), cell, **data)
    for subj, obj, data in G.edges.data():
        dot.edge(
            sha256(subj.encode()).hexdigest(), sha256(obj.encode()).hexdigest(), **data
        )
    return dot
