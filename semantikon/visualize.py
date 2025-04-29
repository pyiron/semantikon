from rdflib import RDF, URIRef, Literal, BNode
from graphviz import Digraph
from string import Template


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
    if ":" in label:
        return {"prefix": dict(graph.namespaces())[label.split(":")[0]]} | kwargs
    else:
        return kwargs


def _get_data(graph):
    data_dict = {}
    edge_list = []
    for subj, value in graph.subject_objects(RDF.value):
        label = _short_label(graph, subj)
        data_dict[label] = _get_value(graph, label, value=str(value.toPython()))

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
    colors = {"prefix": "lightblue", "value": "lightgreen"}
    html = Template(
        """<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">
        $rows
        </table>>"""
    )
    rows = f"<tr><td align='center'>{tag}</td></tr>"
    for key, value in kwargs.items():
        color = colors.get(key, "white")
        rows += (
            f'<tr><td bgcolor="{color}"><font point-size="9">{value}</font></td></tr>'
        )
    return html.substitute(rows=rows)


def visualize(graph):
    dot = Digraph(comment="RDF Graph", format="png")
    dot.attr("node", shape="none", margin="0")
    data_dict, edge_list = _get_data(graph)
    for key, value in data_dict.items():
        if len(value) == 0:
            dot.node(key.replace(":", "_"), _to_node(key))
        else:
            dot.node(key.replace(":", "_"), _to_node(key, **value))
    for edges in edge_list:
        dot.edge(edges[0], edges[1], label=edges[2])

    return dot
