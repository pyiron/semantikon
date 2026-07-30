from hashlib import sha256
from html import escape
from typing import cast

import networkx as nx
from graphviz import Digraph
from rdflib import OWL, RDF, RDFS, Graph, Literal, URIRef
from rdflib.query import ResultRow

from semantikon.ontology import _literal_to_constant

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
        for row in graph.query(query.replace("R_TYPE", rest)):
            subj, pred, obj = cast(ResultRow, row)
            yield subj, pred, obj, style


def _rename_predicate(pred: str) -> str:
    edge_dict = {
        "bfo:0000051": "bfo:has_part",
        "bfo:0000063": "bfo:precedes",
        "iao:0000235": "iao:denoted_by",
        "obi:0001927": "obi:specifies_values_of",
        "pmdco:0000006": "pmdco:has_value",
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
        "pmdco:has_value": "darkred",
        "ro:has_participant": "darkorange",
        "ro:concretizes": "darkcyan",
    }
    return edge_dict.get(pred, "black")


def _node_key(term: URIRef | Literal, owner: URIRef, graph: Graph) -> str:
    """
    Drawing-unique identity for a term.

    Literals have no identity of their own -- two constants of ``2`` are the
    same RDF literal -- so a literal is keyed on the class whose restriction
    points at it. That gives one value box per constant node instead of one
    shared box per distinct value.
    """
    if isinstance(term, Literal):
        return graph.qname(owner) + "-value"
    return graph.qname(term)


def _node_text(term: URIRef | Literal, graph: Graph) -> str:
    """Text drawn inside a term's box."""
    if isinstance(term, Literal):
        return repr(_literal_to_constant(term))
    return graph.qname(term)


def _get_parent_class(comp: URIRef, graph: Graph) -> str:
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


def _get_node_color(comp: URIRef, graph: Graph) -> str:
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
    return not len(list(graph.subjects(OWL.hasValue, term))) > 0


def _rdflib_to_nx(graph: Graph) -> nx.DiGraph:
    G = nx.DiGraph()
    for subj, pred, obj, style in _get_triples(graph):
        keys = []
        for part in [subj, obj]:
            key = _node_key(part, subj, graph)
            keys.append(key)
            if key in G.nodes:
                continue
            G.add_node(
                key,
                fillcolor=_get_node_color(part, graph),
                style="filled" if _is_class(part, graph) else "filled,rounded,dashed",
                shape="box",
                parent_class=_get_parent_class(part, graph),
                text=_node_text(part, graph),
            )
        label = _rename_predicate(graph.qname(pred))
        color = _color_predicate(label)
        G.add_edge(
            *keys,
            label=label,
            style=style,
            color=color,
            fontcolor=color,
        )
    return G


def _to_node(text: str, parent_class: str) -> str:
    translation = {
        "pmdco:0000011": "workflow_node",
        "obi:0001933": "value_specification",
        "pmdco:0000066": "input_assignment",
        "pmdco:0000067": "output_assignment",
        "iao:0000591": "software_method",
    }
    rows = '<<table border="0" cellborder="0" cellspacing="0">'
    # Constant values are arbitrary user data, so escape: an unescaped `<` or
    # `&` makes the whole HTML-like label malformed and graphviz refuses it.
    rows += f"<tr><td align='center'><U>{escape(text)}</U></td></tr>"
    if len(parent_class) > 0:
        subtitle = translation.get(parent_class, parent_class) + " / " + parent_class
        rows += f"<tr><td><I>{escape(subtitle)}</I></td></tr>"
    rows += "</table>>"
    return rows


def visualize_recipe(graph: Graph) -> Digraph:
    G = _rdflib_to_nx(graph)
    dot = Digraph()
    for node, data in G.nodes.data():
        cell = _to_node(data.pop("text"), data.pop("parent_class"))
        dot.node(sha256(node.encode()).hexdigest(), cell, **data)
    for subj, obj, data in G.edges.data():
        dot.edge(
            sha256(subj.encode()).hexdigest(), sha256(obj.encode()).hexdigest(), **data
        )
    return dot
