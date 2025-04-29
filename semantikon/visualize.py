from rdflib import RDF, URIRef, Literal, BNode, Graph
from graphviz import Digraph


def short_label(graph, node):
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


def get_value(graph, label, **kwargs):
    if ":" in label:
        return {"prefix": dict(graph.namespaces())[label.split(":")[0]]} | kwargs
    else:
        return kwargs


def get_data(graph):
    data_dict = {}
    edge_list = []
    for subj, value in graph.subject_objects(RDF.value):
        label = short_label(graph, subj)
        data_dict[label] = get_value(graph, label, value=str(value.toPython()))
    
    for subj, pred, obj in graph:
        if pred == RDF.value:
            continue
        edges = []
        for tag in [subj, obj]:
            label = short_label(graph, tag)
            if label not in data_dict:
                data_dict[label] = get_value(graph, label)
            edges.append(label.replace(":", "_"))
        edges.append(short_label(graph, pred))
        edge_list.append(edges)
    return data_dict, edge_list


def visualize(graph):
    dot = Digraph(comment="RDF Graph", format='png')
    dot.attr('node', shape='record')
    data_dict, edge_list = get_data(graph)
    for key, value in data_dict.items():
        if len(value) == 0:
            dot.node(key.replace(":", "_"), key)
        else:
            dot.node(key.replace(":", "_"), "{" + " | ".join([key] + [f"{kk}: {vv}" for kk, vv in value.items()]) + "}")
    for edges in edge_list:
        dot.edge(edges[0], edges[1], label=edges[2])

    return dot
