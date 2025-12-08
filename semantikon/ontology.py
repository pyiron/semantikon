import uuid
from collections import defaultdict
from dataclasses import is_dataclass
from string import Template
from typing import Any, Callable, TypeAlias, cast

from owlrl import DeductiveClosure, OWLRL_Semantics
from rdflib import OWL, PROV, RDF, RDFS, SH, BNode, Graph, Literal, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.term import IdentifiedNode

from semantikon.converter import get_function_dict, meta_to_dict
from semantikon.metadata import SemantikonURI
from semantikon.qudt import UnitsDict

IAO: Namespace = Namespace("http://purl.obolibrary.org/obo/IAO_")
QUDT: Namespace = Namespace("http://qudt.org/schema/qudt/")
RO: Namespace = Namespace("http://purl.obolibrary.org/obo/RO_")
BFO: Namespace = Namespace("http://purl.obolibrary.org/obo/BFO_")


class SNS:
    BASE: Namespace = Namespace("http://pyiron.org/ontology/")
    has_part: URIRef = BFO["0000051"]
    is_about: URIRef = IAO["0000136"]
    has_unit: URIRef = QUDT["hasUnit"]
    linksTo: URIRef = BASE["linksTo"]
    precedes: URIRef = BFO["0000063"]
    has_participant: URIRef = RO["0000057"]


class NS:
    PREFIX = "semantikon_parent_prefix"
    TYPE = "semantikon_type"


ud = UnitsDict()

_triple_type: TypeAlias = list[
    tuple[IdentifiedNode | str | None, URIRef, IdentifiedNode | str | None]
]


_rest_type: TypeAlias = tuple[tuple[URIRef, URIRef], ...]


def _translate_has_value(
    value_node: URIRef | str | BNode | None,
    value: Any = None,
    dtype: type | None = None,
    units: str | URIRef | None = None,
    custom_triples: _triple_type | None = None,
    derived_from: IdentifiedNode | str | None = None,
    restrictions: _rest_type | tuple[_rest_type] | None = None,
    ontology=SNS,
) -> _triple_type:
    triples: _triple_type = []
    if value is not None:
        triples.append((value_node, RDF.value, Literal(value)))
    if units is not None:
        unit_uri = _units_to_uri(units)
        triples.append((value_node, ontology.has_unit, unit_uri))
    if custom_triples is not None:
        triples.extend(_align_triples(custom_triples))
    if derived_from is not None:
        triples.append(("self", PROV.wasDerivedFrom, derived_from))
    if restrictions is not None:
        triples.extend(_restriction_to_triple(restrictions))
    return triples


def _units_to_uri(units: str | URIRef) -> URIRef:
    if isinstance(units, URIRef):
        return units
    key = ud[units]
    if key is not None:
        return key
    return URIRef(units)

def _align_triples(triples):
    if isinstance(triples[0], tuple | list):
        assert all(len(t) in (2, 3) for t in triples)
        return list(triples)
    else:
        assert len(triples) in (2, 3)
        return [triples]


def _validate_restriction_format(
    restrictions: _rest_type | tuple[_rest_type],
) -> tuple[_rest_type]:
    if isinstance(restrictions[0][0], URIRef):
        return (cast(_rest_type, restrictions),)
    return cast(tuple[_rest_type], restrictions)


def _get_restriction_type(restriction: tuple[tuple[URIRef, URIRef], ...]) -> str:
    if restriction[0][0].startswith(str(OWL)):
        return "OWL"
    elif restriction[0][0].startswith(str(SH)):
        return "SH"
    raise ValueError(f"Unknown restriction type {restriction}")


def _owl_restriction_to_triple(
    restriction: _rest_type,
) -> list[tuple[BNode | None, URIRef, IdentifiedNode]]:
    label = BNode()
    triples = [(None, RDF.type, label), (label, RDF.type, OWL.Restriction)]
    triples.extend([(label, r[0], r[1]) for r in restriction])
    return triples


def _sh_restriction_to_triple(
    restrictions: _rest_type,
) -> list[tuple[str | None, URIRef, URIRef | str]]:
    label = BNode()
    node = str(restrictions[0][0]) + "Node"
    triples = [
        (None, RDF.type, node),
        (node, RDF.type, SH.NodeShape),
        (node, SH.targetClass, node),
        (node, SH.property, label),
    ]
    triples.extend([(label, r[0], r[1]) for r in restrictions])
    return triples


def _restriction_to_triple(
    restrictions: _rest_type | tuple[_rest_type],
) -> list[tuple[IdentifiedNode | str | None, URIRef, IdentifiedNode | str]]:
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
    triples: list[tuple[IdentifiedNode | str | None, URIRef, IdentifiedNode | str]] = []
    for r in restrictions_collection:
        if _get_restriction_type(r) == "OWL":
            triples.extend(_owl_restriction_to_triple(r))
        else:
            triples.extend(_sh_restriction_to_triple(r))
    return triples


def _parse_triple(
    triples: tuple,
    ns: str | None = None,
    label: str | URIRef | None = None,
) -> tuple:
    """
    Triples given in type hints can be expressed by a tuple of 2 or 3 elements.
    If a triple contains 2 elements, the first element is assumed to be the
    predicate and the second element the object, as semantikon automatically
    adds the argument as the subject. If a triple contains 3 elements, the
    first element is assumed to be the subject, the second element the
    predicate, and the third element the object. Instead, you can also
    indicate the position of the argument by setting it to None. Furthermore,
    if the object is a string and starts with "inputs." or "outputs.", it is
    assumed to be a channel and the namespace is added automatically.
    """
    if len(triples) == 2:
        subj, pred, obj = label, triples[0], triples[1]
    elif len(triples) == 3:
        subj, pred, obj = triples
    else:
        raise ValueError("Triple must have 2 or 3 elements")
    assert pred is not None, "Predicate must not be None"

    def _set_tag(tag, ns=None, label=None):
        if tag is None or tag == "self":
            return label
        elif tag.startswith("inputs.") or tag.startswith("outputs."):
            assert ns is not None, "Namespace must not be None"
            return _dot(ns, tag)
        return tag

    subj = _set_tag(subj, ns, label)
    obj = _set_tag(obj, ns, label)
    return subj, pred, obj


def _inherit_properties(
    graph: Graph, triples_to_cancel: list | None = None, n_max: int = 1000, ontology=SNS
):
    update_query = Template(
        f"""\
    PREFIX ns: <{ontology.BASE}>
    PREFIX prov: <{PROV}>
    PREFIX rdfs: <{RDFS}>
    PREFIX rdf: <{RDF}>
    PREFIX owl: <{OWL}>
    PREFIX bfo: <{BFO}>
    PREFIX ro: <{RO}>

    INSERT {{
        ?subject ?p ?o .
    }}
    WHERE {{
        $line
        ?target ?p ?o .
        FILTER(?p != prov:wasDerivedFrom)
        FILTER(?p != rdfs:label)
        FILTER(?p != rdf:value)
        FILTER(?p != rdf:type)
        FILTER(?p != ro:0000057)  # has_participant
        FILTER(?p != bfo:0000051)  # has_part
        FILTER(?p != ns:linksTo)
        FILTER(?p != owl:sameAs)
    }}
    """
    )
    prov_query = update_query.substitute(line="?subject prov:wasDerivedFrom ?target .")
    link_query = update_query.substitute(line="?target ns:linksTo ?subject .")
    if triples_to_cancel is None:
        triples_to_cancel = []
    n = 0
    for _ in range(n_max):
        for query in [prov_query, link_query]:
            graph.update(query)
            for t in triples_to_cancel:
                if t in graph:
                    graph.remove(t)
        if len(graph) == n:
            break
        n = len(graph)


def _check_missing_triples(graph: Graph) -> list:
    query = """SELECT ?instance ?onProperty ?someValuesFrom WHERE {
        ?restriction a owl:Restriction ;
                     owl:onProperty ?onProperty ;
                     owl:someValuesFrom ?someValuesFrom .

        ?cls owl:equivalentClass ?restriction .
        ?instance a ?cls .

        FILTER NOT EXISTS {
            ?instance ?onProperty ?someValuesFrom .
        }
    }"""
    return list(set(graph.query(query)))


def _check_connections(graph: Graph, strict_typing: bool = False, ontology=SNS) -> list:
    """
    Check if the connections between inputs and outputs are compatible

    Args:
        graph (rdflib.Graph): graph to be validated
        strict_typing (bool): if True, check for strict typing

    Returns:
        (list): list of incompatible connections
    """
    incompatible_types = []
    for out, inp in graph.subject_objects(ontology.linksTo):
        i_type, o_type = [
            [g for g in graph.objects(tag, RDF.type) if g != PROV.Entity]
            for tag in (inp, out)
        ]
        # Exclude any i_type that is an OWL restriction or a subclass of OWL.Restriction
        # This is because we handle restrictions in _check_missing_triples
        i_type_filtered = [
            t
            for t in i_type
            if (
                (t, RDFS.subClassOf, OWL.Restriction) not in graph
                and (t, RDF.type, OWL.Restriction) not in graph
                and (t, RDFS.subClassOf, SH.NodeShape) not in graph
                and (t, RDF.type, SH.NodeShape) not in graph
            )
        ]
        if not strict_typing and (i_type_filtered == [] or o_type == []):
            continue
        diff = set(i_type_filtered).difference(o_type)
        if len(diff) > 0:
            incompatible_types.append((inp, out) + (i_type, o_type))
    return incompatible_types


def _check_units(graph: Graph, ontology=SNS) -> dict[URIRef, list[URIRef]]:
    """
    Check if there are multiple units assigned to the same term

    Args:
        graph (rdflib.Graph): graph to be validated

    Returns:
        (dict): dictionary of terms with multiple units
    """
    term_units = defaultdict(list)
    for items in graph.subject_objects(ontology.has_unit):
        term_units[items[0]].append(items[1])
    return {key: value for key, value in term_units.items() if len(value) > 1}


def validate_values(
    graph: Graph, run_reasoner: bool = True, strict_typing: bool = False, ontology=SNS
) -> dict[str, Any]:
    """
    Validate if all values required by restrictions are present in the graph

    Args:
        graph (rdflib.Graph): graph to be validated
        run_reasoner (bool): if True, run the reasoner
        strict_typing (bool): if True, check for strict typing

    Returns:
        (dict): list of missing triples
    """
    if run_reasoner:
        DeductiveClosure(OWLRL_Semantics).expand(graph)
    return {
        "missing_triples": _check_missing_triples(graph),
        "incompatible_connections": _check_connections(
            graph, strict_typing=strict_typing, ontology=ontology
        ),
        "distinct_units": _check_units(graph, ontology=ontology),
    }


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


def _convert_to_uriref(
    value: SemantikonURI | URIRef | Literal | str | None,
    namespace: Namespace | None = None,
) -> URIRef | Literal | BNode:
    if isinstance(value, SemantikonURI):
        return value.get_instance()
    elif isinstance(value, URIRef | Literal | BNode):
        return value
    elif isinstance(value, str):
        if namespace is not None and not value.lower().startswith("http"):
            return namespace[value]
        return BNode(value.replace(".", "-"))
    else:
        raise TypeError(f"Unsupported type: {type(value)}")


def _function_to_triples(function: Callable, node_label: str, ontology=SNS) -> list:
    f_dict = get_function_dict(function)
    triples = []
    if f_dict.get("uri", None) is not None:
        triples.append((node_label, RDF.type, f_dict["uri"]))
    if f_dict.get("triples", None) is not None:
        for t in _align_triples(f_dict["triples"]):
            triples.append(_parse_triple(t, ns=node_label, label=node_label))
    used = f_dict.get("used")
    if used is not None:
        if not isinstance(used, (list, tuple)):
            used = [used]
        for uu in used:
            triples.append((node_label, PROV.used, uu))
    identifier = ".".join([f_dict["module"], f_dict["qualname"], f_dict["version"]])
    triples.append((identifier, ontology.is_about, node_label))
    triples.append((identifier, RDF.type, IAO["0000030"]))
    return triples


def _parse_channel(
    channel_dict: dict, channel_label: str, edge_dict: dict, ontology=SNS
):
    triples: _triple_type = []
    if "type_hint" in channel_dict:
        channel_dict.update(meta_to_dict(channel_dict["type_hint"]))
    triples.append((channel_label, RDF.type, PROV.Entity))
    if channel_dict.get("uri", None) is not None:
        triples.append((channel_label, RDF.type, channel_dict["uri"]))
    value_node = str(edge_dict.get(*2 * [channel_label])) + ".value"
    triples.append((channel_label, ontology.has_participant, value_node))
    triples.extend(
        _translate_has_value(
            value_node=value_node,
            value=channel_dict.get("value", None),
            dtype=channel_dict.get("dtype", None),
            units=channel_dict.get("units", None),
            custom_triples=channel_dict.get("triples", None),
            derived_from=channel_dict.get("derived_from", None),
            restrictions=channel_dict.get("restrictions", None),
            ontology=ontology,
        )
    )
    triples.append(
        (
            channel_label.split(f".{channel_dict[NS.TYPE]}.")[0],
            ontology.has_part,
            channel_label,
        )
    )
    return [
        _parse_triple(t, ns=channel_dict[NS.PREFIX], label=channel_label)
        for t in triples
    ]


def _remove_us(*arg) -> str:
    s = ".".join(arg)
    return ".".join(t.split("__")[-1] for t in s.split("."))


def _get_all_edge_dict(data):
    edges = {}
    for e in data:
        if all(["inputs." in ee for ee in e]):
            edges[e[0]] = e[1]
        else:
            edges[e[1]] = e[0]
    return edges


def _order_edge_dict(data):
    for key, value in data.items():
        if value in data:
            data[key] = data[value]
    return data


def _get_full_edge_dict(data):
    edges = _get_all_edge_dict(data)
    edges = _order_edge_dict(edges)
    return edges


def _get_edge_dict(edges: list) -> dict:
    d = {_remove_us(edge[1]): _remove_us(edge[0]) for edge in edges}
    assert len(d) == len(edges), f"Duplicate keys in edge list: {edges}"
    return d


def _dot(*args) -> str:
    return ".".join([a for a in args if a is not None])


def _edges_to_triples(edges: dict, ontology=SNS) -> list:
    return [
        (upstream, ontology.linksTo, downstream)
        for downstream, upstream in edges.items()
    ]


def _get_precedes(
    edge_dict: dict[str, str], ontology=SNS
) -> list[tuple[str, URIRef, str]]:
    triples = []
    for dest, prov in edge_dict.items():
        if min(len(dest.split(".")), len(prov.split("."))) < 3:
            continue
        dest_node, dest_io, _ = dest.rsplit(".", 2)
        prov_node, prov_io, _ = prov.rsplit(".", 2)
        if dest_io == "inputs" and prov_io == "outputs":
            triples.append((prov_node, SNS.precedes, dest_node))
    return triples


def _parse_workflow(
    node_dict: dict,
    channel_dict: dict,
    edge_list: list,
    ontology=SNS,
) -> list:
    full_edge_dict = _get_full_edge_dict(edge_list)
    triples = [
        triple
        for label, content in channel_dict.items()
        for triple in _parse_channel(content, label, full_edge_dict, ontology)
    ]
    triples.extend(_edges_to_triples(_get_edge_dict(edge_list), ontology=ontology))
    triples.extend(_get_precedes(_get_edge_dict(edge_list), ontology=ontology))

    for key, node in node_dict.items():
        triples.append((key, RDF.type, PROV.Activity))
        if "function" in node:
            triples.extend(_function_to_triples(node["function"], key, ontology))
        if "." in key:
            triples.append((".".join(key.split(".")[:-1]), ontology.has_part, key))
    return triples


def _parse_cancel(wf_channels: dict, namespace: Namespace | None = None) -> list:
    triples = []
    for n_label, channel_dict in wf_channels.items():
        if "extra" not in channel_dict or "cancel" not in channel_dict["extra"]:
            continue
        cancel = channel_dict["extra"]["cancel"]
        assert isinstance(cancel, list | tuple)
        assert len(cancel) > 0
        if not isinstance(cancel[0], list | tuple):
            cancel = [cancel]
        for c in cancel:
            triples.append(_parse_triple(c, label=n_label))
    return [
        tuple([_convert_to_uriref(tt, namespace=namespace) for tt in t])
        for t in triples
    ]


def _translate_dataclass(
    io_port: list[URIRef | str | BNode],
    unique_io_port: str,
    value: Any = None,
    dtype: type | None = None,
    units: str | URIRef | None = None,
    ontology=SNS,
) -> _triple_type:
    value_node = unique_io_port + ".value"
    triples: _triple_type = [
        (io, ontology.has_participant, value_node) for io in io_port
    ]
    for k, v in dtype.__dict__.items():
        if isinstance(v, type) and is_dataclass(v):
            triples.extend(
                _translate_dataclass(
                    io_port=io_port,
                    unique_io_port=_dot(unique_io_port, k),
                    value=getattr(value, k, None),
                    dtype=v,
                    ontology=ontology,
                )
            )
    for k, v in dtype.__annotations__.items():
        metadata = meta_to_dict(v)
        triples.append(
            (_dot(unique_io_port, k) + ".value", RDFS.subClassOf, value_node)
        )
        for io in io_port:
            triples.append(
                (io, ontology.has_participant, _dot(unique_io_port, k) + ".value")
            )
        triples.extend(
            _translate_has_value(
                value_node=_dot(unique_io_port, k) + ".value",
                value=getattr(value, k, None),
                dtype=metadata["dtype"],
                units=metadata.get("units", None),
                ontology=ontology,
            )
        )
    return triples


def _triples_to_knowledge_graph(
    triples: list, graph: Graph | None = None, namespace: Namespace | None = None
) -> Graph:
    if graph is None:
        graph = Graph()
    for triple in triples:
        triple_to_add = []
        for t in triple:
            if t is None:
                break
            triple_to_add.append(_convert_to_uriref(t, namespace=namespace))
            if isinstance(t, SemantikonURI):
                graph.add((t.get_instance(), RDF.type, t.get_class()))
        else:
            graph.add(tuple(triple_to_add))
    return graph


def extract_dataclass(
    graph: Graph, namespace: Namespace | None = None, ontology=SNS
) -> Graph:
    triples = []
    for subj, obj in graph.subject_objects(RDF.value):
        obj = obj.toPython()
        if not is_dataclass(obj):
            continue
        tag = str(subj).rsplit("-value")[0]
        triples.extend(
            _translate_dataclass(
                io_port=list(graph.subjects(ontology.has_participant, subj)),
                unique_io_port=tag,
                value=obj,
                dtype=type(obj),
            )
        )
    return _triples_to_knowledge_graph(triples, namespace=namespace, graph=graph)


def get_knowledge_graph(
    wf_dict: dict,
    graph: Graph | None = None,
    inherit_properties: bool = True,
    ontology=SNS,
    append_missing_items: bool = True,
    use_uuid: bool = False,
    namespace: Namespace | None = None,
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
    node_dict, channel_dict, edge_list = serialize_data(wf_dict, use_uuid=use_uuid)
    triples = _parse_workflow(node_dict, channel_dict, edge_list, ontology=ontology)
    graph = _triples_to_knowledge_graph(triples, graph=graph, namespace=namespace)
    if inherit_properties:
        triples_to_cancel = _parse_cancel(channel_dict, namespace=namespace)
        _inherit_properties(graph, triples_to_cancel, ontology=ontology)
    if append_missing_items:
        graph = _append_missing_items(graph)
    graph.bind("qudt", str(QUDT))
    graph.bind("unit", "http://qudt.org/vocab/unit/")
    graph.bind("sns", str(ontology.BASE))
    graph.bind("prov", str(PROV))
    graph.bind("iao", str(IAO))
    graph.bind("bfo", str(BFO))
    graph.bind("ro", str(RO))
    if namespace is not None:
        graph.bind("ns", str(namespace))
    return graph


def _is_unique(tag, graph):
    return tag not in [h for g in graph.subject_objects(None, None) for h in g]


def _dataclass_to_knowledge_graph(parent, name_space, graph=None, parent_name=None):
    if graph is None:
        graph = Graph()
    for name, obj in vars(parent).items():
        if isinstance(obj, type):  # Check if it's a class
            if _is_unique(name_space[name], graph):
                if parent_name is not None:
                    graph.add(
                        (name_space[name], RDFS.subClassOf, name_space[parent_name])
                    )
                else:
                    graph.add((name_space[name], RDF.type, RDFS.Class))
            else:
                raise ValueError(f"{name} used multiple times")
            _dataclass_to_knowledge_graph(obj, name_space, graph, name)
    return graph


def dataclass_to_knowledge_graph(class_name, name_space):
    """
    Convert a dataclass to a knowledge graph

    Args:
        class_name (dataclass): dataclass to be converted
        name_space (rdflib.Namespace): namespace to be used

    Returns:
        (rdflib.Graph): knowledge graph
    """
    return _dataclass_to_knowledge_graph(
        class_name, name_space, graph=None, parent_name=class_name.__name__
    )


def serialize_data(
    wf_dict: dict, prefix: str | None = None, use_uuid: bool = False
) -> tuple[dict, dict, list]:
    """
    Serialize a nested workflow dictionary into a knowledge graph

    Args:
        wf_dict (dict): workflow dictionary
        prefix (str): prefix to be used for the nodes

    Returns:
        (tuple[dict, dict, list]): node_dict, channel_dict, edge_list
    """
    edge_list = []
    channel_dict = {}
    if prefix is None:
        prefix = wf_dict["label"]
        if use_uuid:
            prefix = f"{prefix}-{uuid.uuid4()}"
    node_dict = {
        prefix: {
            key: value
            for key, value in wf_dict.items()
            if key not in ["nodes", "edges"]
        }
    }
    for io_ in ["inputs", "outputs"]:
        for key, channel in wf_dict[io_].items():
            channel_label = _remove_us(prefix, io_, key)
            assert NS.PREFIX not in channel, f"{NS.PREFIX} already set"
            assert NS.TYPE not in channel, f"{NS.TYPE} already set"
            channel_dict[channel_label] = channel | {
                NS.PREFIX: prefix,
                NS.TYPE: io_,
            }
    for key, node in wf_dict.get("nodes", {}).items():
        child_node, child_channel, child_edges = serialize_data(
            node, prefix=_dot(prefix, key)
        )
        node_dict.update(child_node)
        edge_list.extend(child_edges)
        channel_dict.update(child_channel)
    for args in wf_dict.get("edges", []):
        edge_list.append([_remove_us(prefix, a) for a in args])
    return node_dict, channel_dict, edge_list


def _bundle_restrictions(g: Graph) -> list[BNode]:
    """
    Extract all OWL restriction BNodes from a graph.

    Args:
        g (Graph): RDF graph to search for restrictions

    Returns:
        (list[BNode]): list of BNodes that are OWL restrictions
    """
    return list(g.subjects(RDF.type, OWL.Restriction))


def _to_owl_restriction(
    pred: URIRef,
    target_class: URIRef,
    restriction_type: URIRef = OWL.someValuesFrom,
) -> Graph:
    g = Graph()
    restriction_node = BNode()

    # Build the restriction
    g.add((restriction_node, RDF.type, OWL.Restriction))
    g.add((restriction_node, OWL.onProperty, pred))
    g.add((restriction_node, restriction_type, target_class))

    return g


def _to_intersection(source_class: URIRef, list_items: list[URIRef]) -> Graph:
    g = Graph()
    intersection_node = BNode()
    list_head = BNode()
    g.add((intersection_node, RDF.type, OWL.Class))
    Collection(g, list_head, list_items)
    g.add((intersection_node, OWL.intersectionOf, list_head))

    # Describe source class
    g.add((source_class, RDF.type, OWL.Class))
    g.add((source_class, OWL.equivalentClass, intersection_node))

    return g
