from typing import TypeAlias, Any
import warnings

from rdflib import Graph, Literal, RDF, RDFS, URIRef, OWL, PROV, Namespace
from dataclasses import is_dataclass
from semantikon.converter import meta_to_dict, get_function_dict


class PNS:
    BASE = Namespace("http://pyiron.org/ontology/")
    hasNode = BASE["hasNode"]
    hasSourceFunction = BASE["hasSourceFunction"]
    hasUnits = BASE["hasUnits"]
    inheritsPropertiesFrom = BASE["inheritsPropertiesFrom"]
    inputOf = BASE["inputOf"]
    outputOf = BASE["outputOf"]
    hasValue = BASE["hasValue"]


def _translate_has_value(
    label: URIRef,
    tag: str,
    value: Any = None,
    dtype: type | None = None,
    units: URIRef | None = None,
    parent: URIRef | None = None,
    ontology=PNS,
) -> Graph:
    tag_uri = URIRef(tag + ".value")
    triples = [(label, ontology.hasValue, tag_uri)]
    if is_dataclass(dtype):
        warnings.warn(
            "semantikon_class is experimental - triples may change in the future",
            FutureWarning,
        )
        for k, v in dtype.__dict__.items():
            if isinstance(v, type) and is_dataclass(v):
                triples.extend(
                    _translate_has_value(
                        label=label,
                        tag=tag + "." + k,
                        value=getattr(value, k, None),
                        dtype=v,
                        parent=tag_uri,
                        ontology=ontology,
                    )
                )
        for k, v in dtype.__annotations__.items():
            metadata = meta_to_dict(v)
            triples.extend(
                _translate_has_value(
                    label=label,
                    tag=tag + "." + k,
                    value=getattr(value, k, None),
                    dtype=metadata["dtype"],
                    units=metadata.get("units", None),
                    parent=tag_uri,
                    ontology=ontology,
                )
            )
    else:
        if parent is not None:
            triples.append((tag_uri, RDFS.subClassOf, parent))
        if value is not None:
            triples.append((tag_uri, RDF.value, Literal(value)))
        if units is not None:
            triples.append((tag_uri, ontology.hasUnits, URIRef(units)))
    return triples


def _get_triples_from_restrictions(data: dict) -> list:
    triples = []
    if data.get("restrictions", None) is not None:
        triples = _restriction_to_triple(data["restrictions"])
    if data.get("triples", None) is not None:
        if isinstance(data["triples"][0], tuple | list):
            triples.extend(list(data["triples"]))
        else:
            triples.extend([data["triples"]])
    return triples


_rest_type: TypeAlias = tuple[tuple[URIRef, URIRef], ...]


def _validate_restriction_format(
    restrictions: _rest_type | tuple[_rest_type] | list[_rest_type],
) -> tuple[_rest_type]:
    if not all(isinstance(r, tuple) for r in restrictions):
        raise ValueError("Restrictions must be tuples of URIRefs")
    elif all(isinstance(rr, URIRef) for r in restrictions for rr in r):
        return (restrictions,)
    elif all(isinstance(rrr, URIRef) for r in restrictions for rr in r for rrr in rr):
        return restrictions
    else:
        raise ValueError("Restrictions must be tuples of URIRefs")


def _restriction_to_triple(
    restrictions: _rest_type | tuple[_rest_type] | list[_rest_type],
) -> list[tuple[URIRef | None, URIRef, URIRef]]:
    """
    Convert restrictions to triples

    Args:
        restrictions (tuple): tuple of restrictions

    Returns:
        (list): list of triples

    In the semantikon notation, restrictions are given in the format:

    >>> restrictions = (
    >>>     (OWL.onProperty, EX.HasSomething),
    >>>     (OWL.someValuesFrom, EX.Something)
    >>> )

    This tuple is internally converted to the triples:

    >>> (
    >>>     (EX.HasSomethingRestriction, RDF.type, OWL.Restriction),
    >>>     (EX.HasSomethingRestriction, OWL.onProperty, EX.HasSomething),
    >>>     (EX.HasSomethingRestriction, OWL.someValuesFrom, EX.Something),
    >>>     (my_object, RDFS.subClassOf, EX.HasSomethingRestriction)
    >>> )
    """
    restrictions_collection = _validate_restriction_format(restrictions)
    triples: list[tuple[URIRef | None, URIRef, URIRef]] = []
    for r in restrictions_collection:
        label = r[0][1] + "Restriction"
        triples.append((label, RDF.type, OWL.Restriction))
        for rr in r:
            triples.append((label, rr[0], rr[1]))
        triples.append((None, RDF.type, label))
    return triples


def _parse_triple(
    triples: tuple,
    ns: str,
    label: URIRef | None = None,
) -> tuple:
    if len(triples) == 2:
        subj, pred, obj = label, triples[0], triples[1]
    elif len(triples) == 3:
        subj, pred, obj = triples
    else:
        raise ValueError("Triple must have 2 or 3 elements")
    assert pred is not None, "Predicate must not be None"
    if subj is None:
        subj = label
    if obj is None:
        obj = label
    if obj.startswith("inputs.") or obj.startswith("outputs."):
        obj = ns + "." + obj
    if not isinstance(obj, URIRef):
        obj = URIRef(obj)
    return subj, pred, obj


def _inherit_properties(graph: Graph, n: int | None = None, ontology=PNS):
    update_query = (
        f"PREFIX ns: <{ontology.BASE}>",
        f"PREFIX rdfs: <{RDFS}>",
        f"PREFIX rdf: <{RDF}>",
        "",
        "INSERT {",
        "    ?subject ?p ?o .",
        "}",
        "WHERE {",
        "    ?subject ns:inheritsPropertiesFrom ?target .",
        "    ?target ?p ?o .",
        "    FILTER(?p != ns:inheritsPropertiesFrom)",
        "    FILTER(?p != rdfs:label)",
        "    FILTER(?p != rdf:value)",
        "    FILTER(?p != ns:hasValue)",
        "    FILTER(?p != rdf:type)",
        "}",
    )
    if n is None:
        n = len(list(graph.triples((None, ontology.inheritsPropertiesFrom, None))))
    for _ in range(n):
        graph.update("\n".join(update_query))


def validate_values(graph: Graph) -> list:
    """
    Validate if all values required by restrictions are present in the graph

    Args:
        graph (rdflib.Graph): graph to be validated

    Returns:
        (list): list of missing triples
    """
    missing_triples = []
    for restrictions in graph.subjects(RDF.type, OWL.Restriction):
        on_property = graph.value(restrictions, OWL.onProperty)
        some_values_from = graph.value(restrictions, OWL.someValuesFrom)
        if on_property and some_values_from:
            for cls in graph.subjects(OWL.equivalentClass, restrictions):
                for instance in graph.subjects(RDF.type, cls):
                    if not (instance, on_property, some_values_from) in graph:
                        missing_triples.append(
                            (instance, on_property, some_values_from)
                        )
    return missing_triples


def _append_missing_items(graph: Graph) -> Graph:
    """
    This function makes sure that all properties defined in the descriptions
    become valid.
    """
    for p, o in zip(
        [OWL.onProperty, OWL.someValuesFrom, OWL.allValuesFrom],
        [RDF.Property, OWL.Class, OWL.Class],
    ):
        for obj in graph.objects(None, p):
            triple = (obj, RDF.type, o)
            if triple not in graph:
                graph.add(triple)
    return graph


def _convert_to_uriref(value):
    if isinstance(value, URIRef) or isinstance(value, Literal):
        return value  # Already a URIRef
    elif isinstance(value, str):
        return URIRef(value)  # Convert string to URIRef
    else:
        raise TypeError(f"Unsupported type: {type(value)}")


def _nodes_to_triples(nodes: dict, edge_dict: dict, prefix: str, ontology=PNS) -> list:
    triples = []
    for n_label, node in nodes.items():
        node_label = prefix + "." + n_label
        f_dict = get_function_dict(node["function"])
        if f_dict.get("uri", None) is not None:
            triples.append((node_label, RDF.type, f_dict["uri"]))
        for t in _get_triples_from_restrictions(f_dict):
            triples.append(_parse_triple(t, ns=node_label, label=URIRef(node_label)))
        triples.append((node_label, RDF.type, PROV.Activity))
        triples.append((prefix, ontology.hasNode, node_label))
        triples.append((node_label, ontology.hasSourceFunction, f_dict["label"]))
        for io_ in ["inputs", "outputs"]:
            for key, port in node[io_].items():
                if "type_hint" in port:
                    port.update(meta_to_dict(port["type_hint"]))
                node_label = prefix + "." + n_label
                channel_label = URIRef(f"{node_label}.{io_}.{key}")
                triples.append((channel_label, RDFS.label, Literal(str(channel_label))))
                triples.append((channel_label, RDF.type, PROV.Entity))
                if port.get("uri", None) is not None:
                    triples.append((channel_label, RDF.type, port["uri"]))
                if io_ == "inputs":
                    triples.append((channel_label, ontology.inputOf, node_label))
                elif io_ == "outputs":
                    triples.append((channel_label, ontology.outputOf, node_label))
                tag = edge_dict.get(f"{n_label}.{io_}.{key}", f"{n_label}.{io_}.{key}")
                triples.extend(
                    _translate_has_value(
                        label=channel_label,
                        tag=f"{prefix}.{tag}",
                        value=port.get("value", None),
                        dtype=port.get("dtype", None),
                        units=port.get("units", None),
                        ontology=ontology,
                    )
                )
                for t in _get_triples_from_restrictions(port):
                    triples.append(_parse_triple(t, ns=node_label, label=channel_label))
    return triples


def _get_edge_dict(edges: list) -> dict:
    d = {edge[1]: edge[0] for edge in edges}
    assert len(d) == len(edges), f"Duplicate keys in edge list: {edges}"
    return d


def _edges_to_triples(edges: list, prefix: str, ontology=PNS) -> list:
    return [
        (f"{prefix}.{inp}", ontology.inheritsPropertiesFrom, f"{prefix}.{out}")
        for inp, out in edges.items()
    ]


def get_knowledge_graph(
    wf_dict: dict,
    graph: Graph | None = None,
    inherit_properties: bool = True,
    ontology=PNS,
    append_missing_items: bool = True,
) -> Graph:
    """
    Generate RDF graph from a dictionary containing workflow information

    Args:
        wf_dict (dict): dictionary containing workflow information
        graph (rdflib.Graph): graph to be updated
        inherit_properties (bool): if True, properties are inherited

    Returns:
        (rdflib.Graph): graph containing workflow information
    """
    if graph is None:
        graph = Graph()
    triples = []
    workflow_label = wf_dict.pop("label")
    triples.append((workflow_label, RDFS.label, Literal(workflow_label)))
    edge_dict = _get_edge_dict(wf_dict["data_edges"])
    triples.extend(_edges_to_triples(edge_dict, workflow_label, ontology))
    triples.extend(
        _nodes_to_triples(wf_dict["nodes"], edge_dict, workflow_label, ontology)
    )
    for triple in triples:
        if any(t is None for t in triple):
            continue
        converted_triples = (_convert_to_uriref(t) for t in triple)
        graph.add(converted_triples)
    if inherit_properties:
        _inherit_properties(graph, ontology=ontology)
    if append_missing_items:
        graph = _append_missing_items(graph)
    return graph
