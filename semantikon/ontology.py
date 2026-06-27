from __future__ import annotations

import copy
import json
import unicodedata
import warnings
from dataclasses import asdict, dataclass, fields, is_dataclass
from functools import cache, cached_property
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, TypeAlias, cast

import bagofholding
import flowrep as fr
import networkx as nx
from owlrl import DeductiveClosure, RDFS_Semantics
from pyiron_snippets import retrieve
from pyshacl import validate
from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import SH
from rdflib.term import IdentifiedNode

from semantikon.converter import (
    get_function_dict,
    meta_to_dict,
    parse_input_args,
    parse_output_args,
)
from semantikon.flowrep_dict import (
    annotation_to_type_hint,
    annotation_to_type_metadata,
    dict_to_nodedata,
)
from semantikon.metadata import SemantikonURI
from semantikon.qudt import UnitsDict

IAO: Namespace = Namespace("http://purl.obolibrary.org/obo/IAO_")
NFDI: Namespace = Namespace("https://nfdi.fiz-karlsruhe.de/ontology/NFDI_")
QUDT: Namespace = Namespace("http://qudt.org/schema/qudt/")
RO: Namespace = Namespace("http://purl.obolibrary.org/obo/RO_")
BFO: Namespace = Namespace("http://purl.obolibrary.org/obo/BFO_")
EDAM: Namespace = Namespace("http://edamontology.org/")
OBI: Namespace = Namespace("http://purl.obolibrary.org/obo/OBI_")
PMD: Namespace = Namespace("https://w3id.org/pmd/co/PMD_")
SCHEMA: Namespace = Namespace("http://schema.org/")
STATO: Namespace = Namespace("http://purl.obolibrary.org/obo/STATO_")
BASE: Namespace = Namespace("http://pyiron.org/ontology/")


@dataclass(frozen=True)
class SNS:
    has_part: URIRef = BFO["0000051"]
    part_of: URIRef = BFO["0000050"]
    has_participant: URIRef = RO["0000057"]
    concretizes: URIRef = RO["0000059"]
    input_assignment: URIRef = PMD["0000066"]
    executes: URIRef = STATO["0000102"]
    output_assignment: URIRef = PMD["0000067"]
    precedes: URIRef = BFO["0000063"]
    workflow_node: URIRef = PMD["0000011"]
    continuant: URIRef = BFO["0000002"]
    value_specification: URIRef = OBI["0001933"]
    specifies_value_of: URIRef = OBI["0001927"]
    derives_from: URIRef = RO["0001000"]
    workflow_function: URIRef = PMD["0000010"]
    textual_entity: URIRef = IAO["0000300"]
    denoted_by: URIRef = IAO["0000235"]
    identifier: URIRef = IAO["0020000"]
    is_about: URIRef = IAO["0000136"]
    input_specification: URIRef = PMD["0000014"]
    output_specification: URIRef = PMD["0000015"]
    has_parameter_position: URIRef = PMD["0001857"]
    has_default_literal_value: URIRef = PMD["0001877"]
    has_constraint: URIRef = BASE["has_constraint"]
    has_value: URIRef = PMD["0000006"]
    has_url: URIRef = NFDI["0001008"]
    hdf5: URIRef = EDAM["format_3590"]
    version_number: URIRef = IAO["0000129"]
    import_path: URIRef = PMD["0000101"]
    function_name: URIRef = PMD["0000100"]
    file_data_item: URIRef = NFDI["0000027"]
    local_identifier: URIRef = PMD["0000128"]


ud = UnitsDict()

_triple_type: TypeAlias = list[
    tuple[IdentifiedNode | str | None, URIRef, IdentifiedNode | str | None]
]


_rest_type: TypeAlias = tuple[tuple[URIRef, URIRef], ...]


def _units_to_uri(units: str | URIRef) -> URIRef:
    if isinstance(units, URIRef):
        return units
    key = ud[units]
    if key is not None:
        return key
    return URIRef(units)


class SemantikonDiGraph(nx.DiGraph):
    @cached_property
    def t_ns(self):
        h = (
            "W" + _get_graph_hash(self, with_global_inputs=False)[:8]
            if self.graph["prefix"] is None
            else self.graph["prefix"]
        )
        return Namespace(BASE + h + "_")

    @cached_property
    def a_ns(self):
        h = _get_graph_hash(self, with_global_inputs=True)
        return Namespace(BASE + h + "_")

    def get_a_node(self, node_name: str) -> IdentifiedNode:
        return self.a_ns[node_name]

    @cache
    def _get_data_node(self, io: str) -> str:
        while True:
            candidate = [
                c for c in self.predecessors(io) if self.nodes[c]["step"] != "node"
            ]
            assert len(candidate) <= 1
            if len(candidate) == 0:
                return f"{io}_data"
            io = candidate[0]

    def append_hash(
        self,
        node: str,
        hash_value: str,
        label: str | None = None,
    ):
        """
        Propagates a hash value through the descendants of a given node in a
        directed graph.

        This function iteratively traverses the descendants of the graph and
        appends a hash value to each descendant node. The hash value is
        updated based on the label of each node. Optionally, it can remove
        specific data (e.g., "value") from the nodes.

        Parameters:
            node (str): The starting node from which the hash propagation begins.
            hash_value (str): The initial hash value to propagate through the
                descendants.
            label (str | None, optional): A label to use for hash computation.
                If not provided, the label is derived from the node's data
                (e.g., "label" or "arg"). Defaults to None.

        Notes:
            - The function uses an iterative approach to avoid recursion,
                making it suitable for graphs with deep hierarchies.
            - The hash value for each node is updated in the format:
                `parent_hash@child_label`.

        """
        # Use a stack to keep track of nodes to process
        stack = [(node, hash_value, label)]

        while stack:
            current_node, current_hash, current_label = stack.pop()

            for child in self.successors(current_node):
                if self.nodes[child]["step"] == "node":
                    continue

                # Determine the label for this specific child
                child_label = current_label
                if child_label is None:
                    child_label = self.nodes[child].get(
                        "label", self.nodes[child]["arg"]
                    )

                # Update the hash for the child node
                self.nodes[child]["hash"] = current_hash + f"@{child_label}"

                # Add the child to the stack for further processing
                stack.append((child, current_hash, child_label))

    def get_hash_dict(self) -> dict[str, str]:
        """
        Get a dictionary mapping node names to their hash values.

        Returns:
            dict[str, str]: A dictionary where keys are node names and values
                are their corresponding hash values.
        """
        hash_dict = {}
        for _, data in self.nodes.data():
            if "hash" in data and "value" in data:
                hash_dict[data["hash"]] = data["value"]
        return hash_dict


def _inherit_properties(graph: Graph, n_max: int = 1000):
    query = f"""\
    PREFIX rdfs: <{RDFS}>
    PREFIX rdf: <{RDF}>
    PREFIX owl: <{OWL}>
    PREFIX ro: <{RO}>
    PREFIX pmdco: <{PMD}>
    PREFIX obi: <{OBI}>
    INSERT {{
        ?subject ?p ?o .
    }}
    WHERE {{
        ?subject ro:0001000 ?target .
        ?target ?p ?o .
        FILTER(?p != ro:0001000)
        FILTER(?p != rdfs:label)
        FILTER(?p != pmdco:0000128)
        FILTER(?p != pmdco:0000006)
        FILTER(?p != rdf:type)
        FILTER(?p != owl:sameAs)
        FILTER(?p != obi:0001927)
    }}
    """
    n = 0
    for _ in range(n_max):
        graph.update(query)
        if len(graph) == n:
            break
        n = len(graph)


def _extract_shacl_shapes(input_graph: Graph) -> Graph:
    """
    Extract all SHACL NodeShapes and the full subgraph of constraints
    reachable from them.
    """
    output_graph = _get_bound_graph()
    visited = set()

    def copy_subgraph(node):
        if node in visited:
            return
        visited.add(node)

        for p, o in input_graph.predicate_objects(node):
            output_graph.add((node, p, o))
            # Recurse into blank nodes and referenced shapes
            if isinstance(o, BNode) or (isinstance(o, URIRef) and o.startswith(SH)):
                copy_subgraph(o)

    for shape in input_graph.subjects(RDF.type, SH.NodeShape):
        copy_subgraph(shape)

    return output_graph


def validate_values(
    graph: Graph | dict | fr.schemas.DagData | fr.schemas.WorkflowRecipe,
    run_reasoner: bool = True,
    copy_graph: bool = True,
    strict_typing: bool = True,
) -> tuple:
    """
    Validate if all values required by restrictions are present in the graph

    Args:
        graph (Graph|dict |fr.schemas.DagData|fr.schemas.WorkflowRecipe): input RDF graph, or
            something coercible to one via ``get_knowledge_graph``.
        run_reasoner (bool): if True, run OWL RL reasoner before validation
        copy_graph (bool): if True, work on a copy of the graph. When the graph
            is not copied and reasoner is run, the input graph is modified by
            inferred triples.
        strict_typing (bool): if True, enforce strict typing, i.e. if the
            URI/units of the input of a node are defined, the output of the
            preceding node must have the same URI/units

    Returns:
        (tuple): validation result and message from pyshacl
    """
    if not isinstance(graph, Graph):
        graph = get_knowledge_graph(graph)

    g = copy.deepcopy(graph) if copy_graph and run_reasoner else graph
    excluded = []
    if not strict_typing:
        excluded = _get_undefined_connections(g, "qudt:hasUnit")
        excluded.extend(_get_undefined_connections(g, "obi:0001927"))
    shacl = owl_restrictions_to_shacl(g, excluded_nodes=excluded)
    shacl += _extract_shacl_shapes(g)
    if run_reasoner:
        DeductiveClosure(RDFS_Semantics).expand(g)
        _inherit_properties(g)
    return validate(g, shacl_graph=shacl)


def _get_bound_graph(*args, **kwargs):
    graph = Graph(*args, **kwargs)
    graph.bind("qudt", str(QUDT))
    graph.bind("unit", "http://qudt.org/vocab/unit/")
    graph.bind("sns", str(BASE))
    graph.bind("iao", str(IAO))
    graph.bind("bfo", str(BFO))
    graph.bind("obi", str(OBI))
    graph.bind("ro", str(RO))
    graph.bind("pmdco", str(PMD))
    graph.bind("schema", str(SCHEMA))
    graph.bind("stato", str(STATO))
    graph.bind("nfdi", str(NFDI))
    graph.bind("edam", str(EDAM))
    return graph


def _import_pmdco(pmdco_uri: str) -> Graph:
    g = Graph()
    bnode = BNode("import_pmdco")
    g.add((bnode, RDF.type, OWL.Ontology))
    g.add(
        (
            bnode,
            OWL.imports,
            URIRef(pmdco_uri),
        )
    )
    return g


def _remove_data(graph: Graph) -> Graph:
    graph.update("""DELETE {
            ?data_node pmdco:0000006 ?value .
        }
        WHERE {
            ?data_node pmdco:0000006 ?value .
            ?data_node iao:0000235 ?hash_node .
            ?hash_node a iao:0020000 .
            ?hash_node pmdco:0000006 ?hash .
        }
    """)
    return graph


def _check_consistency_of_digraph(G: SemantikonDiGraph):
    for node, data in G.nodes.data():
        if "derived_from" not in data:
            continue
        expected_input = node.rsplit("-outputs-", 1)[
            0
        ] + f"-{data['derived_from']}".replace(".", "-")
        if expected_input not in G.nodes:
            raise ValueError(
                f"Node '{node}' is derived from '{data['derived_from']}' but"
                f" expected input '{expected_input}' not found in the graph."
            )
        if "uri" not in data and "uri" in G.nodes[expected_input]:
            warnings.warn(
                f"Node '{node}' is derived from '{data['derived_from']}' which"
                " has a URI defined, but the node itself does not have a URI."
                f" '{node}' remains without a URI.",
                UserWarning,
            )


def get_knowledge_graph(
    wf_dict: dict | fr.schemas.DagData | fr.schemas.WorkflowRecipe,
    include_t_box: bool = True,
    include_a_box: bool = True,
    hash_data: bool = True,
    remove_data: bool = False,
    extract_dataclasses: bool = False,
    prefix: str | None = None,
    store_data: bool = False,
    file_name: str | None = None,
    pmdco_uri: str = "https://w3id.org/pmd/co/3.0.0",
) -> Graph:
    """
    Generate RDF graph from workflow information

    Args:
        wf_dict (fr.schemas.DagData|fr.schemas.WorkflowRecipe): ``flowrep``
            object containing workflow information. Passing a ``dict`` is
            deprecated and will be removed in a future version.
        include_t_box (bool): if True, include T-Box information
        include_a_box (bool): if True, include A-Box information
        hash_data (bool): if True, compute and include hash values for data nodes
        remove_data (bool): if True, remove data values after hashing
        extract_dataclasses (bool): if True, extract dataclass information into the graph
        prefix (str | None): prefix to use for the workflow namespace.
            If None, a hash-based prefix is generated.

    Returns:
        (rdflib.Graph): graph containing workflow information
    """
    if isinstance(wf_dict, dict):
        warnings.warn(
            "Passing a dict to 'get_knowledge_graph' is deprecated and will be removed in a future version. "
            "Please pass a 'flowrep' object instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        wf_dict = dict_to_nodedata(wf_dict)
    if isinstance(wf_dict, fr.schemas.WorkflowRecipe):
        wf_dict = fr.schemas.DagData.from_recipe(wf_dict)
    elif isinstance(wf_dict, fr.schemas.DagData):
        pass
    else:
        raise TypeError(
            f"Invalid input type. Expected dict, flowrep {fr.schemas.DagData.__name__!r}, or "
            f"flowrep {fr.schemas.WorkflowRecipe.__name__!r}, but got {type(wf_dict)}."
        )

    G = serialize_and_convert_to_networkx(wf_dict, hash_data=hash_data, prefix=prefix)
    _check_consistency_of_digraph(G)
    graph = _get_bound_graph()
    graph += _import_pmdco(pmdco_uri=pmdco_uri)
    if include_t_box:
        graph += _nx_to_kg(G, t_box=True)
    if include_a_box:
        graph += _nx_to_kg(G, t_box=False)
    if extract_dataclasses:
        graph += extract_dataclass(
            graph=graph, include_t_box=include_t_box, include_a_box=include_a_box
        )
    if store_data:
        if file_name is not None and not file_name.endswith(".h5"):
            file_name += ".h5"
        _store_data(
            graph, file_name=file_name if file_name else f"{_get_graph_hash(G)}.h5"
        )
    if remove_data:
        graph = _remove_data(graph)
    return graph


def _store_data(graph: Graph, file_name: str | Path):
    query = """SELECT DISTINCT ?data_node ?hash ?value WHERE {
        ?data_node pmdco:0000006 ?value .
        ?data_node iao:0000235 ?hash_node .
        ?hash_node a iao:0020000 .
        ?hash_node pmdco:0000006 ?hash .
    }"""
    file_path = str(Path(file_name).absolute().as_uri())
    file_path_id = sha256(file_path.encode("utf-8")).hexdigest()
    file_data_item = BNode(f"file_{file_path_id}")
    data_dict = {}
    for n, h, v in graph.query(query):
        data_dict[h.toPython()] = v.toPython()
        if (file_data_item, RDF.type, SNS.file_data_item) not in graph:
            graph.add((file_data_item, RDF.type, SNS.file_data_item))
            graph.add((file_data_item, SNS.has_url, Literal(file_path)))
            data_format_spec = BNode(f"filefmt_{file_path_id}")
            graph.add((file_data_item, SNS.has_part, data_format_spec))
            graph.add((data_format_spec, RDF.type, SNS.hdf5))
        file_part_id = sha256(str(n).encode("utf-8")).hexdigest()
        file_part = BNode(f"filepart_{file_part_id}")
        graph.add((file_data_item, SNS.has_part, file_part))
        graph.add((file_part, SNS.has_url, Literal(f"object/{h}")))
        graph.add((file_part, SNS.is_about, n))
    bagofholding.H5Bag.save(data_dict, file_name)


def load_data(file_name: str, object_location: str | None = None) -> dict:
    bag = bagofholding.H5Bag(file_name)
    p = f"object/{object_location}" if object_location else "object"
    return bag.load(p)


def function_to_knowledge_graph(function: Callable) -> Graph:
    """
    Generate RDF graph from a Python function

    Converts a Python function into a knowledge graph representation using RDF.
    The graph includes information about the function's inputs, outputs, and metadata.

    Args:
        function (Callable): Python function to convert into a knowledge graph

    Returns:
        (rdflib.Graph): graph containing the function's semantic representation
    """
    output_args = parse_output_args(function)
    if not isinstance(output_args, tuple):
        output_args = (output_args,)
    input_args = []
    for arg, data in parse_input_args(function).items():
        input_args.append({"arg": arg} | data)
    data = get_function_dict(function)
    f_node = BASE["-".join([data["module"], data["qualname"], data["version"]])]
    return _function_to_graph(
        f_node,
        data=data,
        input_args=input_args,
        output_args=output_args,
        uri=data.get("uri"),
    )


def _function_to_graph(
    f_node: URIRef,
    data: dict,
    input_args: list[dict] | tuple[dict, ...],
    output_args: list[dict] | tuple[dict, ...],
    uri: URIRef | None = None,
) -> Graph:
    """
    Converts a function's metadata into an RDF graph representation.

    Args:
        f_node (URIRef): The URI reference for the function node.
        data (dict): A dictionary containing metadata about the function.
                     Expected keys:
                     - "qualname" (str): The qualified name of the function.
                     - "docstring" (str, optional): The docstring of the function.
        input_args (list[dict]): A list of dictionaries representing input arguments.
        output_args (list[dict]): A list of dictionaries representing output arguments.
        uri (URIRef | None, optional): The URI of the function, if available.

    Returns:
        Graph: An RDF graph representing the function and its metadata.
    """
    g = _get_bound_graph()
    g.add((f_node, RDF.type, SNS.workflow_function))
    f_name = BASE[data["qualname"] + "_function_name"]
    g.add((f_name, RDF.type, SNS.function_name))
    g.add((f_name, SNS.has_value, Literal(data["qualname"])))
    g.add((f_node, SNS.denoted_by, f_name))
    g.add(
        (
            f_name,
            RDFS.label,
            Literal(f"Function name '{data['qualname']}'"),
        )
    )
    if data.get("docstring", "") != "":
        docstring = URIRef(f_node + "_docstring")
        g.add((docstring, RDF.type, SNS.textual_entity))
        g.add((docstring, SNS.has_value, Literal(data["docstring"])))
        g.add((docstring, SNS.is_about, f_node))
    if uri is not None:
        assert isinstance(uri, URIRef)
        instance_iri = URIRef(f"{f_node}_instance")
        g.add((f_node, SNS.is_about, instance_iri))
        g.add((instance_iri, RDF.type, uri))
    if data.get("hash", "") != "":
        hash_bnode = URIRef(f_node + "_hash")
        g.add((f_node, SNS.denoted_by, hash_bnode))
        g.add((hash_bnode, RDF.type, SNS.identifier))
        g.add((hash_bnode, SNS.has_value, Literal(data["hash"])))
    if data.get("module", "") != "":
        module = BASE[data["module"].replace(".", "_")]
        g.add((f_node, SNS.denoted_by, module))
        g.add((module, RDF.type, SNS.import_path))
        g.add((module, SNS.has_value, Literal(data["module"])))
    for io, io_args in zip(["input", "output"], [input_args, output_args]):
        for ii, arg in enumerate(io_args):
            if "label" in arg:
                arg_name = arg["label"]
            elif "arg" in arg:
                arg_name = arg["arg"]
            else:
                arg_name = f"output_{ii}"
            arg_node = URIRef("_".join([f_node, io, arg_name]))
            if io == "input":
                g.add((arg_node, RDF.type, SNS.input_specification))
            else:
                g.add((arg_node, RDF.type, SNS.output_specification))
            g.add((arg_node, SNS.local_identifier, Literal(arg_name)))
            g.add((f_node, SNS.has_part, arg_node))
            g.add(
                (arg_node, SNS.has_parameter_position, Literal(arg.get("position", ii)))
            )
            if "default" in arg:
                g.add(
                    (arg_node, SNS.has_default_literal_value, Literal(arg["default"]))
                )
            if "uri" in arg:
                assert isinstance(arg["uri"], URIRef)
                uri_node = BNode(str(arg_node) + "_uri")
                g.add((uri_node, RDF.type, OWL.Restriction))
                g.add((uri_node, OWL.onProperty, SNS.is_about))
                g.add((uri_node, OWL.allValuesFrom, arg["uri"]))
                g.add((arg_node, RDF.type, uri_node))
            if "restrictions" in arg:
                g += _restrictions_to_triples(
                    arg["restrictions"],
                    data_node=arg_node,
                    predicate=SNS.has_constraint,
                )
    return g


def _graph_to_function(graph: Graph, f_node: URIRef) -> dict[str, Any]:
    """
    Extract function metadata from an RDF graph produced by ``_function_to_graph``.

    Args:
        graph (Graph): RDF graph containing function metadata.
        f_node (URIRef): Function node to extract.

    Returns:
        dict[str, Any]: Data payload compatible with ``_function_to_graph``.
    """

    def _to_python(value: IdentifiedNode | Literal | None) -> Any:
        return value.toPython() if isinstance(value, Literal) else value

    def _restriction_pairs(node: IdentifiedNode) -> tuple[tuple[URIRef, Any], ...]:
        return tuple((p, _to_python(o)) for p, o in graph.predicate_objects(node))

    if (f_node, RDF.type, SNS.workflow_function) not in graph:
        raise ValueError(f"Function node {f_node!r} is not present in the graph.")

    function_name_nodes = [
        node
        for node in graph.objects(f_node, SNS.denoted_by)
        if (node, RDF.type, SNS.function_name) in graph
    ]
    if len(function_name_nodes) != 1:
        raise ValueError("Expected exactly one function name node.")
    qualname = graph.value(function_name_nodes[0], SNS.has_value)
    if qualname is None:
        raise ValueError("Function name node is missing `SNS.has_value`.")

    data: dict[str, Any] = {"function": {"qualname": qualname.toPython()}}

    docstring_nodes = [
        node
        for node in graph.subjects(SNS.is_about, f_node)
        if (node, RDF.type, SNS.textual_entity) in graph
    ]
    if len(docstring_nodes) > 1:
        raise ValueError("Expected at most one docstring node.")
    if len(docstring_nodes) == 1:
        docstring = graph.value(docstring_nodes[0], SNS.has_value)
        if docstring is not None:
            data["function"]["docstring"] = docstring.toPython()

    hash_nodes = [
        node
        for node in graph.objects(f_node, SNS.denoted_by)
        if (node, RDF.type, SNS.identifier) in graph
    ]
    if len(hash_nodes) > 1:
        raise ValueError("Expected at most one hash node.")
    if len(hash_nodes) == 1:
        hash_value = graph.value(hash_nodes[0], SNS.has_value)
        if hash_value is not None:
            data["function"]["hash"] = hash_value.toPython()

    module_nodes = [
        node
        for node in graph.objects(f_node, SNS.denoted_by)
        if (node, RDF.type, SNS.import_path) in graph
    ]
    if len(module_nodes) > 1:
        raise ValueError("Expected at most one module node.")
    if len(module_nodes) == 1:
        module = graph.value(module_nodes[0], SNS.has_value)
        if module is not None:
            data["function"]["module"] = module.toPython()

    instance_nodes = [node for node in graph.objects(f_node, SNS.is_about)]
    if len(instance_nodes) > 1:
        raise ValueError("Expected at most one instance node.")
    uri: URIRef | None = None
    if len(instance_nodes) == 1:
        instance_types = list(graph.objects(instance_nodes[0], RDF.type))
        if len(instance_types) > 1:
            raise ValueError("Expected at most one RDF type for the function instance.")
        if len(instance_types) == 1:
            uri = cast(URIRef, instance_types[0])

    input_args: list[dict[str, Any]] = []
    output_args: list[dict[str, Any]] = []

    for arg_node in graph.objects(f_node, SNS.has_part):
        if (arg_node, RDF.type, SNS.input_specification) in graph:
            target = input_args
        elif (arg_node, RDF.type, SNS.output_specification) in graph:
            target = output_args
        else:
            continue

        arg_data: dict[str, Any] = {}
        local_identifier = graph.value(arg_node, SNS.local_identifier)
        if local_identifier is not None:
            arg_data["arg"] = local_identifier.toPython()
        position = graph.value(arg_node, SNS.has_parameter_position)
        if position is not None:
            arg_data["position"] = position.toPython()
        default = graph.value(arg_node, SNS.has_default_literal_value)
        if default is not None:
            arg_data["default"] = default.toPython()

        uri_restrictions = [
            restriction_node
            for restriction_node in graph.objects(arg_node, RDF.type)
            if (restriction_node, RDF.type, OWL.Restriction) in graph
            and (restriction_node, OWL.onProperty, SNS.is_about) in graph
            and graph.value(restriction_node, OWL.allValuesFrom) is not None
        ]
        if len(uri_restrictions) > 1:
            raise ValueError("Expected at most one URI restriction per argument.")
        if len(uri_restrictions) == 1:
            arg_data["uri"] = cast(
                URIRef, graph.value(uri_restrictions[0], OWL.allValuesFrom)
            )

        restrictions = []
        for restriction_node in graph.objects(arg_node, SNS.has_constraint):
            if (restriction_node, RDF.type, OWL.Restriction) in graph:
                pairs = tuple(
                    pair
                    for pair in _restriction_pairs(restriction_node)
                    if pair[0] != RDF.type
                )
            elif (restriction_node, RDF.type, SH.NodeShape) in graph:
                property_shape = graph.value(restriction_node, SH.property)
                if property_shape is None:
                    continue
                pairs = tuple(
                    pair
                    for pair in _restriction_pairs(property_shape)
                    if pair[0] != RDF.type
                )
            else:
                continue
            restrictions.append(pairs)
        if len(restrictions) > 0:
            arg_data["restrictions"] = tuple(restrictions)

        target.append(arg_data)

    input_args.sort(key=lambda d: (d.get("position", 10**9), str(d.get("arg", ""))))
    output_args.sort(key=lambda d: (d.get("position", 10**9), str(d.get("arg", ""))))
    return {
        "f_node": f_node,
        "data": data,
        "input_args": input_args,
        "output_args": output_args,
        "uri": uri,
    }


def _wf_node_to_graph(
    node_name: str,
    data: dict,
    G: SemantikonDiGraph,
    t_box: bool,
) -> Graph:
    g = _get_bound_graph()
    if "function" in data:
        f_node = BASE[data["function"]["identifier"].replace(".", "-")]
        if list(g.triples((f_node, None, None))) == [] and t_box:
            g += _function_to_graph(
                f_node,
                data["function"],
                input_args=[G.nodes[item] for item in G.predecessors(node_name)],
                output_args=[G.nodes[item] for item in G.successors(node_name)],
                uri=data.get("uri"),
            )
    if t_box:
        node = G.t_ns[node_name]
        for io in [G.predecessors(node_name), G.successors(node_name)]:
            for item in io:
                g += _to_owl_restriction(
                    node,
                    SNS.has_part,
                    G.t_ns[item],
                )
        g.add((G.t_ns[node_name], RDFS.subClassOf, SNS.workflow_node))
        if "function" in data:
            g += _to_owl_restriction(
                node,
                SNS.concretizes,
                f_node,
                restriction_type=OWL.hasValue,
            )
        g.add((node, RDFS.label, Literal(node_name)))
        g.add((node, SNS.local_identifier, Literal(node_name.split("-")[-1])))
        if "parent" in data:
            g += _to_owl_restriction(
                G.t_ns[data["parent"]],
                SNS.has_part,
                node,
            )
    else:
        node = G.get_a_node(node_name)
        g.add((node, RDF.type, G.t_ns[node_name]))
        for inp in G.predecessors(node_name):
            g.add((node, SNS.has_part, G.get_a_node(inp)))
        for out in G.successors(node_name):
            g.add((node, SNS.has_part, G.get_a_node(out)))
        if "function" in data:
            g.add((node, SNS.concretizes, f_node))
        if "parent" in data:
            g.add((G.get_a_node(data["parent"]), SNS.has_part, node))
    return g


def _output_is_connected(io: str, G: SemantikonDiGraph) -> bool:
    candidate = list(G.successors(io))
    n_candidates = len(candidate)
    if n_candidates == 0:
        return False
    elif n_candidates == 1:
        if G.nodes[candidate[0]]["step"] == "node":
            return True
        return _output_is_connected(candidate[0], G)
    elif n_candidates == 2 and _is_macro_input(io, G, tuple(candidate)):
        return _output_is_connected(candidate[0], G) and _output_is_connected(
            candidate[1], G
        )
    else:
        return any(_output_is_connected(c, G) for c in candidate)


def _is_macro_input(io: str, G: SemantikonDiGraph, candidates: tuple[str, str]):
    successor_types = {G.nodes[c]["step"] for c in candidates}
    step_type = G.nodes[io]["step"]
    input_edge_predecessors = {"node", "inputs"}
    return step_type == "inputs" and successor_types == input_edge_predecessors


def _input_is_connected(io: str, G: SemantikonDiGraph) -> bool:
    candidate = list(G.predecessors(io))
    n_predecessors = len(candidate)
    if n_predecessors == 0:
        return False
    elif n_predecessors == 1:
        if G.nodes[candidate[0]]["step"] == "node":
            return True
        return _input_is_connected(candidate[0], G)
    elif n_predecessors == 2 and _is_macro_output(io, G, tuple(candidate)):
        return all(
            [
                _input_is_connected(cc, G)
                for cc in candidate
                if G.nodes[cc]["step"] != "node"
            ]
        )
    else:
        predecessor_steps = {c: G.nodes[c]["step"] for c in candidate}
        step_type = G.nodes[io]["step"]
        raise ValueError(
            f"Too many predecessors for {io} ({step_type}): {predecessor_steps}"
        )


def _is_macro_output(io: str, G: SemantikonDiGraph, candidates: tuple[str, str]):
    predecessor_types = {G.nodes[c]["step"] for c in candidates}
    step_type = G.nodes[io]["step"]
    output_edge_predecessors = {"node", "outputs"}
    return step_type == "outputs" and predecessor_types == output_edge_predecessors


def _detect_io_from_str(G: SemantikonDiGraph, seeked_io: str, ref_io: str) -> str:
    assert seeked_io.startswith("inputs") or seeked_io.startswith("outputs")
    main_node = ref_io.replace(".", "-").split("-outputs-")[0].split("-inputs-")[0]
    candidate = (
        G.predecessors(main_node) if "inputs" in seeked_io else G.successors(main_node)
    )
    for io in candidate:
        if io.endswith(seeked_io.replace(".", "-")):
            return G._get_data_node(io=io)
    raise ValueError(f"IO {seeked_io} not found in graph")


def _translate_triples(
    triples: _triple_type,
    node_name: str,
    data_node: URIRef,
    G: SemantikonDiGraph,
    t_box: bool,
) -> Graph:
    def _local_str_to_uriref(t: URIRef | BNode | str | None) -> IdentifiedNode:
        if isinstance(t, SemantikonURI):
            return t.get_instance() if not t_box else t.get_class()
        elif isinstance(t, (URIRef, BNode)):
            return t
        elif t == "self" or t is None:
            return data_node
        else:
            assert isinstance(t, str)
            io = _detect_io_from_str(G=G, seeked_io=t, ref_io=node_name)
            return G.t_ns[io] if t_box else G.get_a_node(io)

    g = _get_bound_graph()
    for triple in triples:
        if len(triple) == 2:
            s = data_node
            p, o = triple
        else:
            s, p, o = triple
        s_n = _local_str_to_uriref(s)
        o_n = _local_str_to_uriref(o)
        if t_box:
            g += _to_owl_restriction(s_n, p, o_n)
        else:
            g.add((s_n, p, o_n))
            for t in [s, o]:
                if isinstance(t, SemantikonURI):
                    g.add((t.get_instance(), RDF.type, t.get_class()))
    return g


def _restrictions_to_triples(
    restrictions: _rest_type, data_node: URIRef, predicate: URIRef | None = None
) -> Graph:
    """
    Converts restrictions into triples for OWL restrictions or SHACL constraints.

    Args:
        restrictions (_rest_type): The restrictions to convert.
        data_node (URIRef): The node to which the restrictions apply.
        predicate (URIRef | None): The predicate to use for OWL restrictions
            (default: RDFS.subClassOf).

    Returns:
        Graph: An RDF graph containing the generated triples.
    """
    g = _get_bound_graph()
    assert isinstance(restrictions, tuple | list)
    assert isinstance(restrictions[0], tuple | list)
    if not isinstance(restrictions[0][0], tuple | list):
        restrictions = cast(_rest_type, (restrictions,))

    for r_set in restrictions:
        # Determine whether the restriction is OWL or SHACL based on the predicates
        is_owl = any(r[0] == OWL.onProperty for r in r_set)
        is_shacl = any(r[0] == SH.path for r in r_set)

        assert (
            is_owl ^ is_shacl
        ), "Unable to determine whether the restrictions are OWL or SHACL."
        if is_owl:
            # Create an OWL Restriction
            if predicate is None:
                predicate = RDFS.subClassOf
            b_node = BNode("rest_" + sha256(str(r_set).encode("utf-8")).hexdigest())
            g.add((data_node, predicate, b_node))
            g.add((b_node, RDF.type, OWL.Restriction))
            for r in r_set:
                g.add((b_node, r[0], r[1]))
        elif is_shacl:
            # Create a SHACL NodeShape
            shape_node = BNode(
                "shape_" + sha256(str(r_set).encode("utf-8")).hexdigest()
            )
            if predicate == SNS.has_constraint:
                g.add((data_node, predicate, shape_node))
            else:
                g.add((shape_node, SH.targetClass, data_node))
            g.add((shape_node, RDF.type, SH.NodeShape))
            sh_property = BNode(str(shape_node) + "_property")
            g.add((shape_node, SH.property, sh_property))

            for r in r_set:
                g.add((sh_property, r[0], r[1]))

    return g


def _wf_input_to_graph(
    node_name: str,
    data: dict,
    G: SemantikonDiGraph,
    t_box: bool,
) -> Graph:
    g = _get_bound_graph()
    units = data.get("units", data.get("unit"))
    if "derived_from" in data:
        raise ValueError(
            f"'derived_from' (defined for the argument '{data['arg']}') is not"
            " supported for inputs."
        )
    if t_box:
        data_node = G.t_ns[G._get_data_node(io=node_name)]
        if _input_is_connected(node_name, G):
            out = list(G.predecessors(node_name))
            assert len(out) <= 1
            if len(out) == 1:
                assert G.nodes[out[0]]["step"] in ["outputs", "inputs"]
                if G.nodes[out[0]]["step"] == "outputs":
                    g += _to_owl_restriction(
                        G.t_ns[out[0]], SNS.has_participant, data_node
                    )
        if units is not None:
            g += _to_owl_restriction(
                base_node=data_node,
                on_property=QUDT.hasUnit,
                target_class=_units_to_uri(units),
                restriction_type=OWL.hasValue,
            )
        if "uri" in data:
            g += _to_owl_restriction(
                data_node,
                SNS.specifies_value_of,
                data["uri"],
                restriction_type=OWL.allValuesFrom,
            )
        if "restrictions" in data:
            g += _restrictions_to_triples(data["restrictions"], data_node=data_node)
    else:
        data_node = G.get_a_node(G._get_data_node(io=node_name))
        if not _input_is_connected(node_name, G):
            if units is not None:
                g.add((data_node, QUDT.hasUnit, _units_to_uri(units)))
            if "uri" in data:
                bnode = URIRef(str(data_node) + "_uri")
                g.add((bnode, RDF.type, data["uri"]))
                g.add((data_node, SNS.specifies_value_of, bnode))
    g += _wf_io_to_graph(
        node_name=node_name,
        data=data,
        data_node=data_node,
        G=G,
        io_assignment=SNS.input_assignment,
        has_specified_io=SNS.has_participant,
        t_box=t_box,
    )
    return g


def _wf_output_to_graph(
    node_name: str,
    data: dict,
    G: SemantikonDiGraph,
    t_box: bool,
) -> Graph:
    g = _get_bound_graph()
    if t_box:
        data_node = G.t_ns[G._get_data_node(io=node_name)]
        if not _output_is_connected(node_name, G):
            if "uri" in data:
                g += _to_owl_restriction(
                    data_node,
                    SNS.specifies_value_of,
                    data["uri"],
                    restriction_type=OWL.allValuesFrom,
                )
            units = data.get("units", data.get("unit"))
            if units is not None:
                g += _to_owl_restriction(
                    base_node=data_node,
                    on_property=QUDT.hasUnit,
                    target_class=_units_to_uri(units),
                    restriction_type=OWL.hasValue,
                )
    else:
        data_node = G.get_a_node(G._get_data_node(io=node_name))
        units = data.get("units", data.get("unit"))
        if units is not None:
            g.add((data_node, QUDT.hasUnit, _units_to_uri(units)))
        if "uri" in data:
            instance = URIRef(str(data_node) + "_uri")
            g.add((instance, RDF.type, data["uri"]))
            g.add((data_node, SNS.specifies_value_of, instance))
    g += _wf_io_to_graph(
        node_name=node_name,
        data=data,
        data_node=data_node,
        G=G,
        io_assignment=SNS.output_assignment,
        has_specified_io=SNS.has_participant,
        t_box=t_box,
    )
    return g


def _wf_io_to_graph(
    node_name: str,
    data: dict,
    data_node: URIRef,
    G: SemantikonDiGraph,
    io_assignment: URIRef,
    has_specified_io: URIRef,
    t_box: bool,
) -> Graph:
    node = G.t_ns[node_name] if t_box else G.get_a_node(node_name)
    g = _get_bound_graph()
    g.add((node, RDFS.label, Literal(node_name)))
    g.add((node, SNS.local_identifier, Literal(node_name.split("-")[-1])))
    if t_box:
        g += _to_owl_restriction(node, has_specified_io, data_node)
        g.add((node, RDFS.subClassOf, io_assignment))
        g.add((data_node, RDFS.subClassOf, SNS.value_specification))
        if "hash" in data:
            g += _to_owl_restriction(data_node, SNS.denoted_by, SNS.identifier)
    else:
        g.add((data_node, RDF.type, G.t_ns[G._get_data_node(io=node_name)]))
        g.add((node, has_specified_io, data_node))
        if "value" in data and g.value(data_node, SNS.has_value) is None:
            g.add((data_node, SNS.has_value, Literal(data["value"])))
        if "hash" in data:
            hash_bnode = G.get_a_node(G._get_data_node(io=node_name) + "_hash")
            g.add((data_node, SNS.denoted_by, hash_bnode))
            g.add((hash_bnode, RDF.type, SNS.identifier))
            g.add((hash_bnode, SNS.has_value, Literal(data["hash"])))
    triples = data.get("triples", [])
    if triples != [] and not isinstance(triples[0], list | tuple):
        triples = [triples]
    if "derived_from" in data:
        triples.append(("self", SNS.derives_from, data["derived_from"]))
    if len(triples) > 0:
        g += _translate_triples(
            triples=triples,
            node_name=node_name,
            data_node=data_node,
            G=G,
            t_box=t_box,
        )
    return g


def _parse_precedes(
    G: SemantikonDiGraph,
    t_box: bool,
) -> Graph:
    g = _get_bound_graph()
    for node in G.nodes.data():
        if node[1]["step"] == "node":
            successors = list(_get_successor_nodes(G, node[0]))
            if t_box:
                for succ in successors:
                    g += _to_owl_restriction(
                        G.t_ns[node[0]],
                        SNS.precedes,
                        G.t_ns[succ],
                    )
            else:
                for succ in successors:
                    g.add(
                        (
                            G.get_a_node(node[0]),
                            SNS.precedes,
                            G.get_a_node(succ),
                        )
                    )
    return g


def _parse_global_io(
    G: SemantikonDiGraph,
    workflow_node: URIRef,
    t_box: bool,
) -> Graph:
    g = _get_bound_graph()
    global_inputs = [
        n for n in G.nodes.data() if G.in_degree(n[0]) == 0 and n[1]["step"] == "inputs"
    ]
    global_outputs = [
        n
        for n in G.nodes.data()
        if G.out_degree(n[0]) == 0 and n[1]["step"] == "outputs"
    ]
    for global_io in [global_inputs, global_outputs]:
        for io in global_io:
            if t_box:
                g += _to_owl_restriction(
                    workflow_node,
                    SNS.has_part,
                    G.t_ns[io[0]],
                )
            else:
                g.add((workflow_node, SNS.has_part, G.get_a_node(io[0])))
    return g


def _nx_to_kg(G: SemantikonDiGraph, t_box: bool) -> Graph:
    g = _get_bound_graph()
    for node_name, data in G.nodes.data():
        data = data.copy()
        step = data.pop("step")
        if t_box:
            g.add((G.t_ns[node_name], RDF.type, OWL.Class))
        else:
            g.add((G.get_a_node(node_name), RDF.type, G.t_ns[node_name]))
        assert step in ["node", "inputs", "outputs"], f"Unknown step: {step}"
        if step == "node":
            g += _wf_node_to_graph(
                node_name=node_name,
                data=data,
                G=G,
                t_box=t_box,
            )
        elif step == "inputs":
            g += _wf_input_to_graph(
                node_name=node_name,
                data=data,
                G=G,
                t_box=t_box,
            )
        else:
            g += _wf_output_to_graph(
                node_name=node_name,
                data=data,
                G=G,
                t_box=t_box,
            )

    g += _parse_precedes(G=G, t_box=t_box)
    return g


class _DataclassTranslator:
    """
    Internal helper class responsible for translating Python dataclasses
    into RDF TBox and ABox statements.

    This class encapsulates the recursive traversal logic and graph
    construction in order to reduce cyclomatic complexity of the public
    API functions.
    """

    def __init__(
        self,
        *,
        include_t_box: bool = True,
        include_a_box: bool = True,
    ) -> None:
        """
        Initialize the translator.

        Args:
            include_t_box: Whether to emit TBox axioms (classes, restrictions).
            include_a_box: Whether to emit ABox assertions (individuals, values).
        """
        self.include_t_box = include_t_box
        self.include_a_box = include_a_box

    def translate(
        self,
        *,
        a_node: URIRef,
        t_node: URIRef,
        value: Any,
        dtype: type,
    ) -> Graph:
        """
        Translate a single dataclass instance and its type into RDF.

        Args:
            a_node: ABox node representing the dataclass instance.
            t_node: TBox node representing the dataclass type.
            value: The Python dataclass instance.
            dtype: The dataclass type.

        Returns:
            An RDF graph containing the generated triples.
        """
        g = _get_bound_graph()

        self._translate_nested_dataclasses(
            graph=g,
            a_node=a_node,
            t_node=t_node,
            value=value,
            dtype=dtype,
        )

        field_metadata = {f.name: f.metadata for f in fields(dtype)}

        for field, annotation in dtype.__annotations__.items():
            metadata = meta_to_dict(annotation)
            metadata.update(field_metadata.get(field, {}))

            t_field = self._to_subkey(t_node, field)
            a_field = self._to_subkey(a_node, field)
            field_value = getattr(value, field, None)

            if self.include_t_box:
                self._emit_tbox(
                    graph=g,
                    parent=t_node,
                    field_node=t_field,
                    metadata=metadata,
                )

            if self.include_a_box:
                self._emit_abox(
                    graph=g,
                    parent=a_node,
                    field_node=a_field,
                    field_class=t_field,
                    metadata=metadata,
                    value=field_value,
                )

        return g

    def _to_subkey(self, node: URIRef, key: str):
        """
        Construct a deterministic sub-node for a dataclass field.

        Args:
            node: Base URIRef
            key: Field name.

        Returns:
            A new URIRef derived from the base node.
        """
        base = str(node).rsplit("_data", 1)[0]
        return node.__class__(f"{base}-{key}")

    def _translate_nested_dataclasses(
        self,
        *,
        graph: Graph,
        a_node: URIRef,
        t_node: URIRef,
        value: Any,
        dtype: type,
    ) -> None:
        """
        Recursively translate nested dataclass fields.

        Args:
            graph: Graph to populate.
            a_node: Parent ABox node.
            t_node: Parent TBox node.
            value: Dataclass instance.
            dtype: Dataclass type.
        """
        for field, field_type in dtype.__dict__.items():
            if isinstance(field_type, type) and is_dataclass(field_type):
                graph += self.translate(
                    a_node=self._to_subkey(a_node, field),
                    t_node=self._to_subkey(t_node, field),
                    value=getattr(value, field, None),
                    dtype=field_type,
                )

    @staticmethod
    def _to_field_label(field_node: URIRef) -> Literal:
        """
        Generate a human-readable label for a dataclass field.

        Args:
            field_node: URIRef representing the field.

        Returns:
            A Literal containing the field label.
        """
        return Literal(str(field_node).rsplit("#", 1)[-1].rsplit("/", 1)[-1])

    def _emit_tbox(
        self,
        *,
        graph: Graph,
        parent: URIRef,
        field_node: URIRef,
        metadata: dict,
    ) -> None:
        """
        Emit TBox axioms for a dataclass field.

        Args:
            graph: Graph to populate.
            parent: Parent class.
            field_node: Field class.
            metadata: Parsed annotation metadata.
        """
        graph += _to_owl_restriction(
            base_node=parent,
            on_property=SNS.has_part,
            target_class=field_node,
        )

        graph.add((field_node, RDFS.subClassOf, SNS.value_specification))
        graph.add((field_node, RDFS.label, self._to_field_label(field_node)))
        graph.add(
            (field_node, SNS.local_identifier, Literal(field_node.split("-")[-1]))
        )

        units = metadata.get("units", metadata.get("unit"))
        if units is not None:
            graph += _to_owl_restriction(
                base_node=field_node,
                on_property=QUDT.hasUnit,
                target_class=_units_to_uri(units),
                restriction_type=OWL.hasValue,
            )

        if "uri" in metadata:
            graph += _to_owl_restriction(
                base_node=field_node,
                on_property=SNS.specifies_value_of,
                target_class=metadata["uri"],
                restriction_type=OWL.allValuesFrom,
            )

    def _emit_abox(
        self,
        *,
        graph: Graph,
        parent: URIRef,
        field_node: URIRef,
        field_class: URIRef,
        metadata: dict,
        value: Any,
    ) -> None:
        """
        Emit ABox assertions for a dataclass field.

        Args:
            graph: Graph to populate.
            field_node: Individual representing the field value.
            field_class: Class of the field.
            metadata: Parsed annotation metadata.
            value: Python value of the field.
        """
        graph.add((parent, SNS.has_part, field_node))
        graph.add((field_node, RDF.type, field_class))
        graph.add((field_node, RDFS.label, self._to_field_label(field_node)))
        graph.add(
            (field_node, SNS.local_identifier, Literal(field_node.split("-")[-1]))
        )

        units = metadata.get("units", metadata.get("unit"))
        if units is not None:
            graph.add((field_node, QUDT.hasUnit, _units_to_uri(units)))

        if "uri" in metadata:
            instance = URIRef(str(field_node) + "_uri")
            graph.add((instance, RDF.type, metadata["uri"]))
            graph.add((field_node, SNS.specifies_value_of, instance))

        if value is not None:
            graph.add((field_node, SNS.has_value, Literal(value)))


def extract_dataclass(
    graph: Graph,
    include_t_box: bool = True,
    include_a_box: bool = True,
) -> Graph:
    """
    Extract dataclass-backed SNS.has_value entries from a graph and translate
    them into RDF TBox and/or ABox triples.

    This function preserves the original public API while delegating
    the translation logic to an internal helper class.

    Args:
        graph: Input RDF graph.
        include_t_box: Whether to include TBox axioms.
        include_a_box: Whether to include ABox assertions.

    Returns:
        A new RDF graph containing the extracted triples.
    """
    out = _get_bound_graph()
    translator = _DataclassTranslator(
        include_t_box=include_t_box,
        include_a_box=include_a_box,
    )

    for subj, obj in graph.subject_objects(SNS.has_value):
        py_value = obj.toPython()
        if not is_dataclass(py_value):
            continue

        t_node = graph.value(subj, RDF.type, any=False)

        out += translator.translate(
            a_node=subj,
            t_node=t_node,
            value=py_value,
            dtype=type(py_value),
        )

    return out


def _get_successor_nodes(G, node_name):
    for out in G.successors(node_name):
        for inp in G.successors(out):
            for node in G.successors(inp):
                yield node


def _infer_workflow_label(recipe: fr.schemas.WorkflowRecipe) -> str:
    if recipe.reference is None:
        return ""
    return recipe.reference.info.fully_qualified_name.rsplit(".", 1)[-1]


def _output_port_label(port: str, outputs: list[str]) -> str:
    if port == "output_0" and len(outputs) == 1 and outputs[0] == "output_0":
        return "output"
    return port


def _port_to_dict(
    *,
    value: Any,
    annotation: Any,
    default: Any = fr.schemas.NOT_DATA,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not isinstance(value, fr.schemas.NotData):
        data["value"] = value
    if type_hint := annotation_to_type_hint(annotation):
        data["dtype"] = type_hint
    if type_metadata := annotation_to_type_metadata(annotation):
        data.update(type_metadata.to_dictionary())
    if not isinstance(default, fr.schemas.NotData):
        data["default"] = default
    return data


def _node_data_to_metadata(
    data: fr.schemas.NodeData,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    function = None
    if isinstance(data, fr.schemas.AtomicData):
        metadata["type"] = "atomic"
        function = data.function
    elif isinstance(data, fr.schemas.DagData):
        metadata["type"] = "workflow"
        if label is not None:
            metadata["label"] = label
        if data.recipe.reference is not None:
            function = retrieve.import_from_string(
                data.recipe.reference.info.fully_qualified_name
            )
    if function is not None:
        if hasattr(function, "_semantikon_metadata"):
            metadata.update(function._semantikon_metadata)
        function_data = get_function_dict(function)
        function_data["identifier"] = ".".join(
            (
                function_data["module"],
                function_data["qualname"],
                function_data["version"],
            )
        )
        metadata["function"] = function_data
    return metadata


def _workflow_to_networkx(
    workflow: fr.schemas.DagData,
    *,
    prefix: str | None = None,
) -> SemantikonDiGraph:
    root_label = _infer_workflow_label(workflow.recipe)
    G = SemantikonDiGraph(prefix=prefix)
    G.name = root_label

    def _add_node(
        node_data: fr.schemas.NodeData,
        node_name: str,
        *,
        parent_name: str | None = None,
        workflow_label: str | None = None,
    ):
        node_attrs = _node_data_to_metadata(node_data, label=workflow_label)
        if parent_name is not None:
            node_attrs["parent"] = parent_name
        G.add_node(node_name, step="node", **node_attrs)

        output_labels = list(node_data.output_ports)
        if len(output_labels) == 1 and output_labels[0] == "output_0":
            output_labels = ["output"]

        for position, (label, port) in enumerate(node_data.input_ports.items()):
            io_name = f"{node_name}-inputs-{label}"
            io_data = _port_to_dict(
                value=port.value,
                annotation=port.annotation,
                default=port.default,
            )
            G.add_node(io_name, step="inputs", arg=label, position=position, **io_data)
            G.add_edge(io_name, node_name)
        for position, (raw_label, port) in enumerate(node_data.output_ports.items()):
            label = output_labels[position] if raw_label == "output_0" else raw_label
            io_name = f"{node_name}-outputs-{label}"
            io_data = _port_to_dict(value=port.value, annotation=port.annotation)
            G.add_node(io_name, step="outputs", arg=label, position=position, **io_data)
            G.add_edge(node_name, io_name)

        if not isinstance(node_data, fr.schemas.DagData):
            return

        recipe = node_data.recipe
        for child_label, child in node_data.nodes.items():
            child_name = f"{node_name}-{child_label}"
            _add_node(
                child,
                child_name,
                parent_name=node_name,
                workflow_label=(
                    child_label if isinstance(child, fr.schemas.DagData) else None
                ),
            )

        child_recipes = recipe.nodes
        for target, source in recipe.input_edges.items():
            G.add_edge(
                f"{node_name}-inputs-{source.port}",
                f"{node_name}-{target.node}-inputs-{target.port}",
            )
        for target, source in recipe.edges.items():
            child_outputs = list(child_recipes[source.node].outputs)
            src_port = _output_port_label(source.port, child_outputs)
            G.add_edge(
                f"{node_name}-{source.node}-outputs-{src_port}",
                f"{node_name}-{target.node}-inputs-{target.port}",
            )
        for target, source in recipe.output_edges.items():
            target_port = _output_port_label(target.port, list(recipe.outputs))
            if isinstance(source, fr.schemas.InputSource):
                G.add_edge(
                    f"{node_name}-inputs-{source.port}",
                    f"{node_name}-outputs-{target_port}",
                )
            else:
                child_outputs = list(child_recipes[source.node].outputs)
                src_port = _output_port_label(source.port, child_outputs)
                G.add_edge(
                    f"{node_name}-{source.node}-outputs-{src_port}",
                    f"{node_name}-outputs-{target_port}",
                )

    _add_node(workflow, root_label, workflow_label=root_label)
    return G


def _get_hashed_node_dict_from_graph(G: SemantikonDiGraph) -> dict[str, dict[str, Any]]:
    hash_dict: dict[str, dict[str, Any]] = {}
    for node in nx.topological_sort(G):
        data = G.nodes[node]
        if data.get("step") != "node":
            for term in ("hash", "value"):
                if term in data:
                    continue
                for predecessor in G.predecessors(node):
                    predecessor_data = G.nodes[predecessor]
                    if term in predecessor_data:
                        data[term] = predecessor_data[term]
                        break
            continue

        hash_dict_tmp: dict[str, Any] = {
            "inputs": {},
            "outputs": [
                G.nodes[out].get("label", out.split("-")[-1])
                for out in G.successors(node)
            ],
            "node": copy.deepcopy(data.get("function")),
        }
        if hash_dict_tmp["node"] is None:
            continue
        hash_dict_tmp["node"]["connected_inputs"] = []
        missing_input = False
        for inp in G.predecessors(node):
            inp_data = G.nodes[inp]
            inp_name = inp.split("-")[-1]
            if "hash" in inp_data:
                hash_dict_tmp["inputs"][inp_name] = inp_data["hash"]
                hash_dict_tmp["node"]["connected_inputs"].append(inp_name)
            elif "value" in inp_data:
                value = inp_data["value"]
                if is_dataclass(value) and not isinstance(value, type):
                    hash_dict_tmp["inputs"][inp_name] = asdict(value)
                else:
                    hash_dict_tmp["inputs"][inp_name] = value
            else:
                missing_input = True
                break
        if missing_input:
            continue
        h = sha256(
            json.dumps(hash_dict_tmp, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for out in G.successors(node):
            G.nodes[out]["hash"] = (
                h + "@" + G.nodes[out].get("label", out.split("-")[-1])
            )
        hash_dict_tmp["hash"] = h
        hash_dict[node] = hash_dict_tmp
    return hash_dict


def serialize_and_convert_to_networkx(
    workflow: dict | fr.schemas.DagData | fr.schemas.WorkflowRecipe,
    hash_data: bool = True,
    prefix: str | None = None,
) -> SemantikonDiGraph:
    """
    Serialize a flowrep workflow into a SemantikonDiGraph, optionally
    hashing node data.

    Args:
        workflow (dict | DagData | WorkflowRecipe): Workflow representation.
        hash_data (bool): Whether to hash node data.
        prefix (str | None): Optional prefix for node names.

    Returns:
        SemantikonDiGraph: The serialized workflow graph.
    """
    if isinstance(workflow, dict):
        workflow = dict_to_nodedata(workflow)
    if isinstance(workflow, fr.schemas.WorkflowRecipe):
        workflow = fr.schemas.DagData.from_recipe(workflow)
    if not isinstance(workflow, fr.schemas.DagData):
        raise TypeError(
            f"Invalid workflow type. Expected dict, flowrep {fr.schemas.DagData.__name__!r}, or "
            f"flowrep {fr.schemas.WorkflowRecipe.__name__!r}, but got {type(workflow)}."
        )

    G = _workflow_to_networkx(workflow, prefix=prefix)
    if hash_data:
        try:
            hashed_dict = _get_hashed_node_dict_from_graph(G)
        except Exception as e:
            raise RuntimeError(
                "Failed to hash workflow data - use only hashable inputs or set hash_data=False"
            ) from e
        for node, data in hashed_dict.items():
            G.append_hash(node, data["hash"])
    return G


def serialize_and_networkx_to_data(G: nx.DiGraph) -> fr.schemas.DagData:
    """
    Convert a NetworkX DiGraph back into flowrep DagData structure.

    This is the inverse of ``serialize_and_convert_to_networkx``.

    Args:
        G (nx.DiGraph): Serialized workflow graph with Semantikon node/edge schema.

    Returns:
        fr.schemas.DagData: The reconstructed workflow data.
    """
    return _networkx_to_dict(G)


def _networkx_to_dict(G: nx.DiGraph) -> fr.schemas.DagData:
    """
    Convert a NetworkX DiGraph into flowrep DagData.

    Args:
        G (nx.DiGraph): Graph to convert, using Semantikon node/edge attributes.

    Returns:
        fr.schemas.DagData: Reconstructed workflow data.
    """

    def _extract_io_data(io_name: str) -> dict[str, Any]:
        data = {}
        node_data = G.nodes[io_name]
        if "value" in node_data:
            data["value"] = node_data["value"]
        if "dtype" in node_data:
            data["dtype"] = node_data["dtype"]
        if "default" in node_data:
            data["default"] = node_data["default"]
        for key in ["uri", "units", "unit", "triples", "derived_from", "restrictions"]:
            if key in node_data:
                data[key] = node_data[key]
        return data

    def _get_function_from_dict(func_dict: dict[str, Any]) -> Any:
        """Try to reconstruct the function from the stored metadata."""
        if not isinstance(func_dict, dict):
            return func_dict
        try:
            module = func_dict.get("module")
            qualname = func_dict.get("qualname")
            if module and qualname:
                fqn = f"{module}.{qualname}"
                return retrieve.import_from_string(fqn)
        except Exception:
            pass
        return None

    def _process_node(node_name: str) -> dict[str, Any]:
        node_data = G.nodes[node_name]
        node_type = node_data.get("type", "atomic")

        result: dict[str, Any] = {"type": node_type}

        if "function" in node_data:
            func_obj = _get_function_from_dict(node_data["function"])
            if func_obj is not None:
                result["function"] = func_obj

        if "label" in node_data:
            result["label"] = node_data["label"]

        input_ports = {}
        output_ports = {}
        for predecessor in G.predecessors(node_name):
            pred_data = G.nodes[predecessor]
            if pred_data.get("step") == "inputs":
                port_name = pred_data["arg"]
                input_ports[port_name] = _extract_io_data(predecessor)
        for successor in G.successors(node_name):
            succ_data = G.nodes[successor]
            if succ_data.get("step") == "outputs":
                port_name = succ_data["arg"]
                output_ports[port_name] = _extract_io_data(successor)

        if input_ports:
            result["inputs"] = input_ports
        if output_ports:
            result["outputs"] = output_ports

        if node_type == "workflow":
            nodes = {}
            edges = []

            for child_label, child_node in G.nodes.items():
                if (
                    child_node.get("step") == "node"
                    and child_node.get("parent") == node_name
                ):
                    child_short_label = child_label.split("-", 1)[1]
                    nodes[child_short_label] = _process_node(child_label)

            for u, v in G.edges:
                u_data = G.nodes[u]
                v_data = G.nodes[v]
                u_step = u_data.get("step")
                v_step = v_data.get("step")

                if u_step == "inputs" and v_step == "inputs":
                    u_port = u_data["arg"]
                    v_parts = v.split("-")
                    if len(v_parts) >= 3:
                        v_child = v_parts[1]
                        v_port = v_data["arg"]
                        edges.append((f"inputs.{u_port}", f"{v_child}.inputs.{v_port}"))
                elif u_step == "outputs" and v_step == "inputs":
                    u_parts = u.split("-")
                    v_parts = v.split("-")
                    if len(u_parts) >= 3 and len(v_parts) >= 3:
                        u_child = u_parts[1]
                        v_child = v_parts[1]
                        u_port = u_data["arg"]
                        v_port = v_data["arg"]
                        edges.append(
                            (
                                f"{u_child}.outputs.{u_port}",
                                f"{v_child}.inputs.{v_port}",
                            )
                        )
                elif u_step == "inputs" and v_step == "outputs" and u != v:
                    u_port = u_data["arg"]
                    v_port = v_data["arg"]
                    edges.append((f"inputs.{u_port}", f"outputs.{v_port}"))
                elif u_step == "outputs" and v_step == "outputs" and u != v:
                    u_parts = u.split("-")
                    if len(u_parts) >= 3:
                        u_child = u_parts[1]
                        u_port = u_data["arg"]
                        v_port = v_data["arg"]
                        edges.append(
                            (f"{u_child}.outputs.{u_port}", f"outputs.{v_port}")
                        )

            if nodes:
                result["nodes"] = nodes
            if edges:
                result["edges"] = edges

        return result

    root_node = G.name
    wf_dict = _process_node(root_node)
    return cast(fr.schemas.DagData, dict_to_nodedata(wf_dict))


def _to_owl_restriction(
    base_node: URIRef | None,
    on_property: URIRef,
    target_class: URIRef,
    restriction_type: URIRef = OWL.someValuesFrom,
) -> Graph:
    g = _get_bound_graph()
    restriction_node = BNode(
        sha256(
            str((base_node, on_property, target_class, restriction_type)).encode(
                "utf-8"
            )
        ).hexdigest()
    )

    # Build the restriction
    g.add((restriction_node, RDF.type, OWL.Restriction))
    g.add((restriction_node, OWL.onProperty, on_property))
    g.add((restriction_node, restriction_type, target_class))
    if base_node is not None:
        g.add((base_node, RDFS.subClassOf, restriction_node))

    return g


class _HashGraph:
    def _normalize(self, obj: Any) -> Any:
        """
        Convert objects into a deterministic, JSON-safe representation.
        """
        if is_dataclass(obj) and not isinstance(obj, type):
            return {k: self._normalize(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: self._normalize(v) for k, v in obj.items()}
        elif isinstance(obj, IdentifiedNode):
            if isinstance(obj, BNode):
                raise TypeError("Blank nodes cannot be normalized for hashing.")
            return {"__type__": "URIRef", "value": self._normalize(str(obj))}
        elif isinstance(obj, (list, tuple)):
            return [self._normalize(v) for v in obj]
        elif isinstance(obj, str):
            return unicodedata.normalize("NFC", obj)
        elif isinstance(obj, (int, float, bool)) or obj is None:
            return obj
        else:
            raise TypeError(
                f"Unsupported type for normalization: {type(obj)} of value {obj}"
            )

    def _canonical_json(self, data: dict) -> str:
        """
        Deterministic JSON serialization.
        """
        return json.dumps(
            self._normalize(data),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def _get_graph_hash(
        self, G: SemantikonDiGraph, with_global_inputs: bool = True
    ) -> str:
        """
        Generate a deterministic hash for a graph, independent of OS,
        Python version, and non-serializable runtime values.
        """
        G_tmp = nx.DiGraph()

        for node in G.nodes:
            attrs = {
                key: value
                for key, value in G.nodes[node].items()
                if key not in {"dtype", "hash", "function", "default", "value"}
            }
            if G.in_degree(node) == 0 and with_global_inputs:
                if "value" in G.nodes[node]:
                    attrs["value"] = G.nodes[node]["value"]
                elif "default" in G.nodes[node]:
                    attrs["value"] = G.nodes[node]["default"]
            G_tmp.add_node(node, canon=self._canonical_json(attrs))
        for u, v in G.edges:
            G_tmp.add_edge(u, v)

        return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
            G_tmp,
            node_attr="canon",
        )


def _get_graph_hash(G: SemantikonDiGraph, with_global_inputs: bool = True) -> str:
    """
    Generate a hash for a NetworkX graph, making sure that data types and
    values (except for the global ones) because they can often not be
    serialized.

    Args:
        G (SemantikonDiGraph): input graph
        with_global_inputs (bool): if True, keep values for global inputs

    Returns:
        (str): hash of the graph
    """
    hasher = _HashGraph()
    return hasher._get_graph_hash(G, with_global_inputs)


def _get_undefined_connections(g, term):
    query = "\n".join(
        [f"PREFIX {key}: <{value}>" for key, value in dict(g.namespaces()).items()]
    )
    query += f"""
    SELECT ?rnode WHERE {{
      ?class rdfs:subClassOf ?rnode .
      ?rnode a owl:Restriction .
      ?rnode owl:onProperty {term} .
      ?instance a ?class .
      FILTER NOT EXISTS {{
        ?instance {term} ?bnode .
      }}
    }}"""
    return [item for items in g.query(query) for item in items]


class _OWLToSHACLConverter:
    def __init__(self, owl_graph: Graph, excluded_nodes: list[BNode] | None = None):
        """
        Initialize the converter with an OWL graph and optional excluded nodes.
        """
        self.owl_graph = owl_graph
        self.excluded_nodes = excluded_nodes

    @staticmethod
    def _new_shacl_graph():
        shacl_graph = _get_bound_graph()
        shacl_graph.bind("sh", str(SH))
        shacl_graph.bind("sns", str(BASE))
        return shacl_graph

    def _iter_supported_restrictions(self):
        """
        Yield (base_class, property, restriction_type, value) tuples
        for supported OWL restrictions.
        """
        for r in self.owl_graph.subjects(RDF.type, OWL.Restriction):
            if self.excluded_nodes is not None and r in self.excluded_nodes:
                continue
            prop = self.owl_graph.value(r, OWL.onProperty)
            if prop is None:
                continue
            for restriction_type in (
                OWL.someValuesFrom,
                OWL.hasValue,
                OWL.allValuesFrom,
            ):
                value = self.owl_graph.value(r, restriction_type)
                if value is None:
                    continue

                for base_cls in self.owl_graph.subjects(RDFS.subClassOf, r):
                    yield base_cls, prop, restriction_type, value

    def _translate_restrictions(self) -> Graph:
        node_shapes: dict[URIRef, BNode] = {}
        shacl_graph = self._new_shacl_graph()
        for base_cls, prop, rtype, value in self._iter_supported_restrictions():

            # One NodeShape per base class
            if not (ns := node_shapes.get(base_cls)):
                ns = BNode()
                node_shapes[base_cls] = ns
                shacl_graph.add((ns, RDF.type, SH.NodeShape))
                shacl_graph.add((ns, SH.targetClass, base_cls))

            ps = BNode()
            shacl_graph.add((ps, RDF.type, SH.PropertyShape))
            shacl_graph.add((ps, SH.path, prop))

            if rtype == OWL.someValuesFrom:
                # Existential restriction:
                # ∃ prop . C  →  qualifiedValueShape + qualifiedMinCount
                qvs = BNode()
                shacl_graph.add((qvs, SH["class"], value))

                shacl_graph.add((ps, SH.qualifiedValueShape, qvs))
                shacl_graph.add((ps, SH.qualifiedMinCount, Literal(1)))

            elif rtype == OWL.hasValue:
                shacl_graph.add((ps, SH.hasValue, value))

            elif rtype == OWL.allValuesFrom:
                # Universal restriction:
                # ∀ prop . C  →  class
                shacl_graph.add((ps, SH["class"], value))

            shacl_graph.add((ns, SH.property, ps))
        return shacl_graph

    def _translate_disjoint_with(self) -> Graph:
        """
        Translate OWL disjointWith axioms into SHACL shapes with sh:not.
        """
        node_shapes: dict[URIRef, BNode] = {}
        shacl_graph = self._new_shacl_graph()
        for cls in self.owl_graph.subjects(RDF.type, OWL.Class):
            disjoints = list(self.owl_graph.objects(cls, OWL.disjointWith))
            if not disjoints:
                continue

            # Create a NodeShape for the class if it doesn't exist
            if not (ns := node_shapes.get(cls)):
                ns = BNode()
                node_shapes[cls] = ns
                shacl_graph.add((ns, RDF.type, SH.NodeShape))
                shacl_graph.add((ns, SH.targetClass, cls))

            # Create a shape that represents the disjoint classes
            disjoint_shape = BNode()
            shacl_graph.add((disjoint_shape, RDF.type, SH.NodeShape))
            for disjoint_cls in disjoints:
                shacl_graph.add((disjoint_shape, SH.targetClass, disjoint_cls))

            # Add a sh:not constraint to the original shape
            shacl_graph.add((ns, SH["not"], disjoint_shape))
        return shacl_graph

    def convert(self) -> Graph:
        """
        Convert the OWL logics into SHACL shapes, including both restrictions
        and disjointness axioms.
        """
        shacl_graph = self._translate_restrictions()
        shacl_graph += self._translate_disjoint_with()
        return shacl_graph


def owl_restrictions_to_shacl(
    owl_graph: Graph, excluded_nodes: list[BNode] | None = None
) -> Graph:
    """
    Convert OWL restrictions in the given graph to SHACL shapes.

    This function is a wrapper around the _OWLToSHACLConverter class.
    """
    converter = _OWLToSHACLConverter(owl_graph, excluded_nodes)
    return converter.convert()
