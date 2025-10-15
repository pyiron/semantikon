from string import Template

from graphviz import Digraph
from rdflib import RDF, BNode, Literal, URIRef

from semantikon.ontology import IAO, SNS

_edge_colors = {
    "rdf:type": "darkblue",
    "iao:0000136": "darkgreen",
    "sns:outputOf": "darkred",
    "sns:inputOf": "darkorange",
    "sns:hasValue": "brown",
    "sns:linksTo": "gray",
    "ro:0000051": "deeppink",
    "prov:wasDerivedFrom": "purple",
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


def _get_data(graph):
    data_dict = {}
    edge_list = []
    for subj, value in graph.subject_objects(RDF.value):
        label = _short_label(graph, subj)
        data_dict[label] = _get_value(graph, label, value=str(value.toPython()))

    for obj in graph.objects(None, RDF.type):
        _add_color(data_dict, graph, obj, "lightyellow")

    for obj in graph.subjects(IAO.isAbout, None):
        _add_color(data_dict, graph, obj, "lightgreen")

    for subj, obj in graph.subject_objects(SNS.outputOf):
        _add_color(data_dict, graph, obj, "lightpink")
        _add_color(data_dict, graph, subj, "lightcyan")

    for subj, obj in graph.subject_objects(SNS.inputOf):
        _add_color(data_dict, graph, obj, "lightpink")
        _add_color(data_dict, graph, subj, "lightskyblue")

    for obj in graph.objects(None, SNS.hasValue):
        _add_color(data_dict, graph, obj, "peachpuff")

    for subj, pred, obj in graph:
        if pred == RDF.value:
            continue
        edges = []
        for tag in [subj, obj]:
            label = _short_label(graph, tag)
            if label not in data_dict:
                data_dict[label] = _get_value(graph, label)
            edges.append(label.replace(":", "_"))
        edges.append(_short_label(graph, pred))
        edge_list.append(edges)
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


def visualize(graph, engine="dot"):
    dot = Digraph(comment="RDF Graph", format="png", engine=engine)
    dot.attr(overlap="false")
    dot.attr(splines="true")
    dot.attr("node", shape="none", margin="0")
    data_dict, edge_list = _get_data(graph)
    for key, value in data_dict.items():
        if len(value) == 0:
            dot.node(key.replace(":", "_"), _to_node(key))
        else:
            dot.node(key.replace(":", "_"), _to_node(key, **value))
    for edges in edge_list:
        color = _edge_colors.get(edges[2], "black")
        dot.edge(edges[0], edges[1], label=edges[2], color=color, fontcolor=color)

    return dot
