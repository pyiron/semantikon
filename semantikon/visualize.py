from string import Template

from graphviz import Digraph
from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, URIRef

from semantikon.ontology import SNS

_edge_colors = {
    "rdf:type": "darkblue",
    "iao:0000136": "darkgreen",
    "bfo:0000051": "darkred",
    "ro:0000057": "brown",
    "sns:linksTo": "gray",
    "bfo:0000063": "deeppink",
    "prov:wasDerivedFrom": "purple",
}

_id_to_tag = {
    "iao:0000136": "iao:is_about",
    "bfo:0000051": "bfo:has_part",
    "ro:0000057": "ro:has_participant",
    "bfo:0000063": "bfo:precedes",
}


def _short_label(graph, node):
    """Use graph's prefixes to shorten URIs nicely."""
    if isinstance(node, URIRef):
        try:
            return graph.qname(node)
        except Exception:
            return str(node)
    elif isinstance(node, BNode):
        return f"_:{str(node)}"
    elif isinstance(node, Literal):
        return f'"{str(node)}"'
    else:
        return str(node)


def _get_value(graph, label, **kwargs):
    prefix = dict(graph.namespaces()).get(label.split(":")[0])
    if prefix is not None:
        return {"prefix": prefix} | kwargs
    else:
        return kwargs


def _add_color(data_dict, graph, tag, color):
    label = _short_label(graph, tag)
    if label not in data_dict:
        data_dict[label] = _get_value(graph, label) | {"bgcolor": color}
    else:
        data_dict[label]["bgcolor"] = color


def _simplify_restrictions(graph):
    """
    Simplify OWL restrictions into dotted triples for cleaner visualization.

    Converts restriction patterns of the form:
      node -> rdf:type -> BNode -> rdf:type -> owl:Restriction
                        -> owl:onProperty -> property
                        -> owl:someValuesFrom -> class
    into simplified dotted triples: (node, property, class)

    Args:
        graph: An RDFLib Graph containing OWL restrictions.

    Returns:
        tuple: (new_graph, dotted_triples) where new_graph has restrictions removed
               and dotted_triples is a list of simplified (subj, pred, obj) tuples.

    Raises:
        AssertionError: If any restriction doesn't have exactly one subject,
                       property, or someValuesFrom object.

    Assumptions:
        - Each OWL restriction is represented as a blank node with exactly one subject,
          one property (owl:onProperty), and one someValuesFrom object.
        - The function removes the original restriction triples from the graph.
    """
    dotted_triples = []
    triples_to_remove = []
    for b_node in graph.subjects(RDF.type, OWL.Restriction):
        subj = list(graph.subjects(RDF.type, b_node))
        assert len(subj) == 1, "Assertion failed: set simplify_restrictions to False"
        pred = list(graph.objects(b_node, OWL.onProperty))
        assert len(pred) == 1, "Assertion failed: set simplify_restrictions to False"
        obj = list(graph.objects(b_node, OWL.someValuesFrom))
        assert len(obj) == 1, "Assertion failed: set simplify_restrictions to False"
        dotted_triples.append((subj[0], pred[0], obj[0]))
        triples_to_remove.extend(
            (
                (subj[0], RDF.type, b_node),
                (b_node, RDF.type, OWL.Restriction),
                (b_node, OWL.onProperty, pred[0]),
                (pred[0], RDF.type, RDF.Property),
                (b_node, OWL.someValuesFrom, obj[0]),
            )
        )

    new_graph = Graph()
    for triple in graph:
        if triple not in triples_to_remove:
            new_graph.add(triple)
    return new_graph, dotted_triples


def _get_data(graph, simplify_restrictions=False):
    dotted_triples = []
    if simplify_restrictions:
        graph, dotted_triples = _simplify_restrictions(graph)
    data_dict = {}
    edge_list = []
    for subj, value in graph.subject_objects(RDF.value):
        label = _short_label(graph, subj)
        data_dict[label] = _get_value(graph, label, value=str(value.toPython()))

    for obj in graph.objects(None, RDF.type):
        _add_color(data_dict, graph, obj, "lightyellow")

    for obj in graph.objects(None, RDFS.subClassOf):
        _add_color(data_dict, graph, obj, "lightyellow")

    for obj in graph.objects(None, SNS.has_part):
        _add_color(data_dict, graph, obj, "lightpink")

    for obj in graph.objects(None, SNS.has_participant):
        _add_color(data_dict, graph, obj, "peachpuff")

    for subj, obj in graph.subject_objects(SNS.is_about):
        _add_color(data_dict, graph, obj, "lightcyan")
        _add_color(data_dict, graph, subj, "lightgreen")

    for style, g in [("solid", graph), ("dotted", dotted_triples)]:
        for subj, pred, obj in g:
            if pred == RDF.value:
                continue
            edge = {"edge": []}
            for tag in [subj, obj]:
                label = _short_label(graph, tag)
                if label not in data_dict:
                    data_dict[label] = _get_value(graph, label)
                edge["edge"].append(label.replace(":", "_"))
            edge["label"] = _short_label(graph, pred)
            edge["style"] = style
            edge_list.append(edge)
    return data_dict, edge_list


def _to_node(tag, **kwargs):
    colors = {"prefix": "green", "value": "red"}
    html = Template(
        """<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">
        $rows
        </table>>"""
    )
    bgcolor = kwargs.pop("bgcolor", "white")
    rows = f"<tr><td align='center' bgcolor='{bgcolor}'>{tag}</td></tr>"
    for key, value in kwargs.items():
        color = colors.get(key, "black")
        rows += f'<tr><td><font point-size="9" color="{color}">{value}</font></td></tr>'
    return html.substitute(rows=rows)


def visualize(graph, engine="dot", simplify_restrictions=False):
    """
    Visualize an RDF graph using Graphviz.

    Args:
        graph: An RDFLib Graph to visualize.
        engine (str): Graphviz layout engine to use (default: "dot").
        simplify_restrictions (bool): If True, simplify OWL restrictions into dotted edges
            for cleaner visualization (default: False).

    Returns:
        Digraph: A graphviz Digraph object representing the visualized graph.
    """
    dot = Digraph(comment="RDF Graph", format="png", engine=engine)
    dot.attr(overlap="false")
    dot.attr(splines="true")
    dot.attr("node", shape="none", margin="0")
    data_dict, edge_list = _get_data(graph, simplify_restrictions=simplify_restrictions)
    for key, value in data_dict.items():
        if len(value) == 0:
            dot.node(key.replace(":", "_"), _to_node(key))
        else:
            dot.node(key.replace(":", "_"), _to_node(key, **value))
    for edges in edge_list:
        color = _edge_colors.get(edges["label"], "black")
        label = _id_to_tag.get(edges["label"], edges["label"])
        dot.edge(
            edges["edge"][0],
            edges["edge"][1],
            label=label,
            color=color,
            fontcolor=color,
            style=edges["style"],
        )
    return dot
