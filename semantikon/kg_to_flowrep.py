from __future__ import annotations

import hashlib
import itertools
from typing import Any, cast

import flowrep as fr
import networkx as nx
from pyiron_snippets import retrieve
from rdflib import OWL, RDF, RDFS, Graph, Literal, URIRef, term
from rdflib.namespace import SH

from semantikon.flowrep_dict import _flowrep_recipe_from_callable
from semantikon.flowrep_to_networkx import (
    IO,
    Input,
    Node,
    Output,
    TNodeData,
    TOutputData,
)
from semantikon.ontology import SNS


def _graph_to_function(graph: Graph, f_node: URIRef) -> dict[str, Any]:
    """
    Extract function metadata from an RDF graph produced by ``_function_to_graph``.

    Args:
        graph (Graph): RDF graph containing function metadata.
        f_node (URIRef): Function node to extract.

    Returns:
        dict[str, Any]: Data payload compatible with ``_function_to_graph``.
    """

    def _to_python(value: term.Node | None) -> Any:
        return value.toPython() if isinstance(value, Literal) else value

    def _restriction_pairs(node: term.IdentifiedNode) -> tuple[tuple[URIRef, Any], ...]:
        return tuple(
            (cast(URIRef, p), _to_python(o)) for p, o in graph.predicate_objects(node)
        )

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

    data: dict[str, Any] = {"qualname": cast(Literal, qualname).toPython()}

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
            data["docstring"] = cast(Literal, docstring).toPython()

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
            data["hash"] = cast(Literal, hash_value).toPython()

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
            data["module"] = cast(Literal, module).toPython()

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
            arg_data["arg"] = cast(Literal, local_identifier).toPython()
        position = graph.value(arg_node, SNS.has_parameter_position)
        if position is not None:
            arg_data["position"] = cast(Literal, position).toPython()
        default = graph.value(arg_node, SNS.has_default_literal_value)
        if default is not None:
            arg_data["default"] = cast(Literal, default).toPython()

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
                    for pair in _restriction_pairs(
                        cast(term.IdentifiedNode, restriction_node)
                    )
                    if pair[0] != RDF.type
                )
            elif (restriction_node, RDF.type, SH.NodeShape) in graph:
                property_shape = graph.value(restriction_node, SH.property)
                if property_shape is None:
                    continue
                pairs = tuple(
                    pair
                    for pair in _restriction_pairs(
                        cast(term.IdentifiedNode, property_shape)
                    )
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


def _networkx_to_flowrep(G: nx.DiGraph) -> fr.schemas.WorkflowRecipe:
    """
    Convert a NetworkX DiGraph into flowrep WorkflowRecipe.

    Args:
        G (nx.DiGraph): Graph to convert, using Semantikon node/edge attributes.

    Returns:
        fr.schemas.WorkflowRecipe: Reconstructed workflow recipe.
    """

    def _get_function_from_dict(func_dict: Any) -> Any:
        """Try to reconstruct the function from the stored metadata."""
        if not isinstance(func_dict, dict):
            return func_dict

        module = func_dict.get("module")
        qualname = func_dict.get("qualname")
        if not module or not qualname:
            raise ValueError(
                f"Cannot reconstruct function; missing 'module'/'qualname' in {func_dict!r}"
            )

        fqn = f"{module}.{qualname}"
        try:
            return retrieve.import_from_string(fqn)
        except Exception as exc:
            raise ImportError(
                f"Failed to import {fqn!r} while reconstructing a workflow from"
                + " a knowledge graph."
            ) from exc

    def _normalize_output_label(label: str, recipe_outputs: list[str]) -> str:
        if label == "output" and recipe_outputs == ["output_0"]:
            return "output_0"
        return label

    def _process_node(node_name: Node) -> fr.schemas.RecipeDiscrimination:
        node_data = G.nodes[node_name]
        node_type = node_data.get("type", "atomic")
        if node_type == "constant":
            output_node = list(G.successors(node_name))
            assert (
                len(output_node) == 1
            ), f"Constant node {node_name} should have one output."
            return fr.schemas.ConstantRecipe(constant=G.nodes[output_node[0]]["value"])
        if "function" not in node_data:
            raise ValueError(f"Node {node_name!r} is missing function metadata.")
        func_obj = _get_function_from_dict(node_data["function"])
        if node_type == "workflow":
            base_recipe = _flowrep_recipe_from_callable(func_obj, node_type="workflow")
            nodes: dict[str, fr.schemas.RecipeDiscrimination] = {}
            input_edges: fr.schemas.InputEdges = {}
            edges: fr.schemas.Edges = {}
            output_edges: fr.schemas.OutputEdges = {}

            # First collect all direct children
            direct_children: dict[str, Node] = {}
            for child_label in G.nodes:
                if isinstance(child_label, Node) and child_label.parent == node_name:
                    # Extract child short label correctly by removing parent prefix
                    direct_children[child_label.name] = child_label
                    nodes[child_label.name] = _process_node(child_label)

            def _find_child_for_io(io_node_name: IO) -> str | None:
                """Find the child node (short label) that owns this IO node."""
                for child_label in direct_children.values():
                    if io_node_name.node == child_label:
                        return child_label.name
                return None

            def _is_direct_io(node_id: Node | IO) -> bool:
                """Check if this is a direct IO of the workflow."""
                if isinstance(node_id, Node):
                    return False
                return node_id.node == node_name

            def _is_child_io(node_id: IO | Node) -> bool:
                """Check if this is an IO of a direct child (and only direct child)."""
                if not isinstance(node_id, IO):
                    return False
                for child_label in direct_children.values():
                    if node_id.node == child_label:
                        return True
                return False

            for u, v in G.edges:
                u_is_direct_io = _is_direct_io(u)
                v_is_direct_io = _is_direct_io(v)
                u_is_child_io = _is_child_io(u)
                v_is_child_io = _is_child_io(v)

                if isinstance(u, Input) and isinstance(v, Input):
                    if u_is_direct_io and v_is_child_io:
                        v_child = _find_child_for_io(v)
                        if v_child is not None:
                            edges_key = fr.schemas.TargetHandle(
                                node=v_child, port=v.port
                            )
                            input_edges[edges_key] = fr.schemas.InputSource(port=u.port)
                elif isinstance(u, Output) and isinstance(v, Input):
                    if u_is_child_io and v_is_child_io:
                        u_child = _find_child_for_io(u)
                        v_child = _find_child_for_io(v)
                        if u_child is not None and v_child is not None:
                            u_port = _normalize_output_label(
                                u.port, nodes[u_child].outputs
                            )
                            edges_key = fr.schemas.TargetHandle(
                                node=v_child, port=v.port
                            )
                            edges[edges_key] = fr.schemas.SourceHandle(
                                node=u_child, port=u_port
                            )
                elif (
                    isinstance(u, Output)
                    and isinstance(v, Output)
                    and u != v
                    and u_is_child_io
                    and v_is_direct_io
                ):
                    u_child = _find_child_for_io(u)
                    if u_child is not None:
                        u_port = _normalize_output_label(u.port, nodes[u_child].outputs)
                        v_port = _normalize_output_label(
                            v.port, list(base_recipe.outputs)
                        )
                        output_edges[fr.schemas.OutputTarget(port=v_port)] = (
                            fr.schemas.SourceHandle(node=u_child, port=u_port)
                        )
            return fr.schemas.WorkflowRecipe(
                inputs=list(base_recipe.inputs),
                outputs=list(base_recipe.outputs),
                nodes=nodes,
                input_edges=input_edges,
                edges=edges,
                output_edges=output_edges,
                reference=base_recipe.reference,
                description=base_recipe.description,
            )
        if node_type == "atomic":
            return _flowrep_recipe_from_callable(func_obj, node_type="atomic")
        raise TypeError(f"Unsupported workflow node type: {node_type!r}")

    root_node = Node(G.name)
    recipe = _process_node(root_node)
    if not isinstance(recipe, fr.schemas.WorkflowRecipe):
        raise TypeError(
            f"Root node {root_node!r} must be a workflow, got {type(recipe).__name__!r}."
        )
    return recipe


def _get_restriction(
    subj: str, pred: URIRef, obj: str, r_type: URIRef = OWL.someValuesFrom
) -> str:
    b_node = "?b_" + hashlib.sha256((subj + obj).encode("utf-8")).hexdigest()[:8]
    return f"""{subj} <{RDFS.subClassOf}> {b_node} .
    {b_node} a <{OWL.Restriction}> .
    {b_node} <{OWL.onProperty}> <{pred}> .
    {b_node} <{r_type}> {obj} ."""


def _get_connection_query(
    subj: URIRef, pred: URIRef, obj: URIRef, r_type: URIRef = OWL.someValuesFrom
) -> str:
    return f"""SELECT ?s ?o WHERE {{
    ?s <{RDFS.subClassOf}> <{subj}> .
    ?o <{RDFS.subClassOf}> <{obj}> .
    {_get_restriction("?s", pred, "?o", r_type=r_type)}
    }}"""


def _identifier(graph: Graph, node: URIRef) -> str:
    local_identifier = graph.value(node, SNS.local_identifier)
    if local_identifier is not None:
        return cast(Literal, local_identifier).toPython()
    label = graph.value(node, RDFS.label)
    if label is not None:
        return cast(Literal, label).toPython()
    return str(node)


def _label(graph: Graph, node: URIRef) -> str:
    label = graph.value(node, RDFS.label)
    if label is not None:
        return cast(Literal, label).toPython()
    return _identifier(graph, node)


def _node_functions(graph: Graph) -> dict[URIRef, URIRef]:
    query = f"""SELECT ?node ?function WHERE {{
        ?node <{RDFS.subClassOf}> <{SNS.workflow_node}> .
        ?node <{RDFS.subClassOf}> ?bnode .
        ?bnode a <{OWL.Restriction}> .
        ?bnode <{OWL.hasValue}> ?function .
        ?bnode <{OWL.onProperty}> <{SNS.concretizes}> .
    }}"""
    return dict(graph.query(query))


def _reorganize_output_edges(graph: nx.DiGraph, node: URIRef, position: dict[Any, int]):
    io_dict: dict[URIRef, URIRef] = {}
    for n in graph.predecessors(node):
        pred = list(graph.predecessors(n))
        assert len(pred) == 1 and pred[0] not in io_dict, f"{pred}, {n}"
        io_dict[pred[0]] = n
    keys = sorted(io_dict.keys(), key=lambda item: position[item])[::-1]
    nodes = [io_dict[k] for k in keys]
    for n in nodes[:-1]:
        graph.remove_edge(n, node)
    graph.add_edges_from(itertools.pairwise(nodes))


def _reorganize_input_edges(graph: nx.DiGraph, node: URIRef, position: dict[Any, int]):
    io_dict: dict[URIRef, URIRef] = {}
    for n in graph.successors(node):
        succ = list(graph.successors(n))
        assert len(succ) == 1 and succ[0] not in io_dict, succ
        io_dict[succ[0]] = n
    node_keys = sorted(io_dict.keys(), key=lambda item: position[item])
    for i, key_one in enumerate(node_keys):
        for key_two in node_keys[i + 1 :]:
            if (key_one, key_two) in graph.edges:
                graph.remove_edge(node, io_dict[key_two])
                graph.add_edge(io_dict[key_one], io_dict[key_two])


def _reconnect_io(graph: nx.DiGraph, node: URIRef):
    outputs = list(graph.predecessors(node))
    inputs = list(graph.successors(node))
    assert len(inputs) > 0 and len(outputs) == 1
    for inp in inputs:
        graph.add_edge(outputs[0], inp)


def _uri_to_node_names(graph: Graph):
    node_graph = nx.DiGraph()
    for parent, child in graph.query(  # type: ignore[misc]
        _get_connection_query(SNS.workflow_node, SNS.has_part, SNS.workflow_node)
    ):
        node_graph.add_edge(parent, child)

    node_dict: dict[URIRef, Node] = {}
    for parent in nx.topological_sort(node_graph):
        parent_name = cast(
            Literal, graph.value(parent, SNS.local_identifier)
        ).toPython()
        node_dict[parent] = node_dict.get(parent, Node(parent_name))
        for child in node_graph.successors(parent):
            assert child not in node_dict
            child_name = cast(
                Literal, graph.value(child, SNS.local_identifier)
            ).toPython()
            node_dict[child] = Node(child_name, parent=node_dict[parent])
    return node_dict


def _uri_to_node_and_io_names(
    graph: Graph, uri_to_node: dict[URIRef, Node], G: nx.DiGraph
) -> dict[URIRef, Node | IO]:
    io_dict: dict[URIRef, IO] = {}
    for uri, node in uri_to_node.items():
        for out in G.successors(uri):
            if out in uri_to_node:  # node; not an IO
                continue
            arg = graph.value(out, SNS.local_identifier)
            io_dict[out] = Output(port=arg.toPython(), node=node)
        for inp in G.predecessors(uri):
            if inp in uri_to_node:  # node; not an IO
                continue
            arg = graph.value(inp, SNS.local_identifier)
            io_dict[inp] = Input(port=arg.toPython(), node=node)
    return io_dict | uri_to_node


def _build_workflow_graph(graph: Graph) -> nx.DiGraph:

    workflow_graph = nx.DiGraph()

    for node, io_node in graph.query(  # type: ignore[misc]
        _get_connection_query(SNS.workflow_node, SNS.has_part, SNS.input_assignment)
    ):
        workflow_graph.add_edge(io_node, node)

    for node, io_node in graph.query(  # type: ignore[misc]
        _get_connection_query(SNS.workflow_node, SNS.has_part, SNS.output_assignment)
    ):
        workflow_graph.add_edge(node, io_node)

    for out_assignment, data_node in graph.query(  # type: ignore[misc]
        _get_connection_query(
            SNS.output_assignment, SNS.has_participant, SNS.value_specification
        )
    ):
        workflow_graph.add_edge(out_assignment, data_node)

    for in_assignment, data_node in graph.query(  # type: ignore[misc]
        _get_connection_query(
            SNS.input_assignment, SNS.has_participant, SNS.value_specification
        )
    ):
        workflow_graph.add_edge(data_node, in_assignment)

    # This is only needed to correctly identify input - input and
    # output - output edges in the workflow graph.
    for parent_node, child_node in graph.query(  # type: ignore[misc]
        _get_connection_query(SNS.workflow_node, SNS.has_part, SNS.workflow_node)
    ):
        workflow_graph.add_edge(parent_node, child_node)
    return workflow_graph


def _append_metadata_to_graph(graph: Graph, workflow_graph: nx.DiGraph) -> None:
    node_to_f = _node_functions(graph)
    for node in workflow_graph.nodes:
        if node not in node_to_f:
            continue
        f_meta = _graph_to_function(graph, node_to_f[node])
        workflow_graph.nodes[node]["function"] = f_meta["data"]
        input_args = {ent.pop("arg"): ent for ent in f_meta["input_args"]}
        output_args = {ent.pop("arg"): ent for ent in f_meta["output_args"]}
        for inp_node in workflow_graph.predecessors(node):
            if a := graph.value(inp_node, SNS.local_identifier):
                if a.toPython() not in input_args:
                    continue
                for key, value in input_args[a.toPython()].items():
                    workflow_graph.nodes[inp_node][key] = value
        for out_node in workflow_graph.successors(node):
            if a := graph.value(out_node, SNS.local_identifier):
                if a.toPython() not in output_args:
                    continue
                for key, value in output_args[a.toPython()].items():
                    workflow_graph.nodes[out_node][key] = value


def _get_node_positions(workflow_graph: nx.DiGraph) -> dict[Node, int]:
    return {
        node: i
        for i, node in enumerate(nx.topological_sort(workflow_graph))
        if isinstance(node, Node)
    }


def _reorganize_workflow_graph(workflow_graph: nx.DiGraph) -> None:
    """
    Reorganize the workflow graph to ensure that data nodes are properly connected
    to their input and output assignments, and that the graph is in a suitable
    format for conversion to a flowrep WorkflowRecipe.

    Args:
        workflow_graph (nx.DiGraph): The workflow graph to reorganize.
    """
    position = _get_node_positions(workflow_graph)
    for node in tuple(workflow_graph.nodes):
        if not isinstance(node, URIRef):
            continue
        if len(list(workflow_graph.predecessors(node))) > 1:
            _reorganize_output_edges(workflow_graph, node, position)
        if len(list(workflow_graph.successors(node))) > 1:
            _reorganize_input_edges(workflow_graph, node, position)
        if all(
            len(list(direction(node))) > 0
            for direction in (workflow_graph.predecessors, workflow_graph.successors)
        ):
            _reconnect_io(workflow_graph, node)

    workflow_graph.remove_nodes_from(
        [node for node in workflow_graph.nodes if isinstance(node, URIRef)]
    )
    workflow_graph.remove_edges_from(
        [
            edge
            for edge in workflow_graph.edges
            if all(isinstance(node, Node) for node in edge)
        ]
    )
    workflow_graph.name = str(next(iter(position.keys())))


def _extract_constant_values_from_kg(
    rdf_graph: Graph,
    workflow_graph: nx.DiGraph,
    uri_to_node_and_io: dict[URIRef, Node | IO],
) -> None:
    """
    Extract constant values from the RDF knowledge graph and add them to
    input nodes in the workflow graph.

    This extracts constant values that are stored as OWL.hasValue restrictions
    on SNS.has_value in the data nodes.

    Args:
        rdf_graph (Graph): The RDF knowledge graph.
        workflow_graph (nx.DiGraph): The workflow graph to modify in-place.
    """
    from semantikon.ontology import _literal_to_constant

    # Query for all data nodes with hasValue restrictions on SNS.has_value
    query = f"""\
    PREFIX owl: <{OWL}>
    PREFIX rdfs: <{RDFS}>

    SELECT ?input_node ?value
    WHERE {{
        ?input_node rdfs:subClassOf <{SNS.input_assignment}> .
        ?i_rest a owl:Restriction .
        ?i_rest owl:onProperty <{SNS.has_participant}> .
        ?i_rest owl:someValuesFrom ?value_node .
        ?input_node rdfs:subClassOf ?i_rest .
        ?value_node rdfs:subClassOf <{SNS.value_specification}> .
        ?value_node rdfs:subClassOf ?v_rest .
        ?v_rest a owl:Restriction .
        ?v_rest owl:onProperty <{SNS.has_value}> .
        ?v_rest owl:hasValue ?value .
    }}"""

    for input_node, value_literal in rdf_graph.query(query):  # type: ignore[misc]
        inp = uri_to_node_and_io.get(input_node)  # type: ignore[arg-type]
        workflow_graph.nodes[inp]["constant_value"] = (
            _literal_to_constant(value_literal)
            if isinstance(value_literal, Literal)
            else value_literal
        )


def _reconstruct_constant_nodes(G: nx.DiGraph) -> None:
    """
    Reconstruct constant nodes in a workflow graph by finding input nodes with
    constant_value attributes and creating the appropriate constant node structure.

    This reverses the _remove_constant operation from flowrep_to_networkx.

    Args:
        G (nx.DiGraph): Workflow graph to modify in-place.
    """

    for node, data in tuple(G.nodes.data()):
        if not isinstance(node, Input) or "constant_value" not in data:
            continue

        for constant_index in itertools.count():
            const_node = Node(
                name=f"{fr.schemas.ConstantRecipe.std_label}_{constant_index}",
                parent=node.node.parent,
            )
            if const_node not in G:
                break

        const_output_name = Output(port="constant", node=const_node)

        # Create the constant node
        const_node_attrs = TNodeData(type="constant")
        G.add_node(const_node, **const_node_attrs.to_attrs())

        # Create the constant output node
        const_output_attrs = TOutputData(
            position=0,
            value=data["constant_value"],
            has_value=True,
            dtype=data.get("dtype", None),
        )
        G.add_node(const_output_name, **const_output_attrs.to_attrs())

        # Record edges to add
        G.add_edge(const_node, const_output_name)
        G.add_edge(const_output_name, node)


def _ensure_workflow_name(
    uri_to_node: dict[URIRef, Node],
    wf_name: str | URIRef | None = None,
) -> URIRef:
    roots = {k: v for k, v in uri_to_node.items() if v.parent is None}
    if len(roots) == 0:
        raise ValueError(
            "No workflow nodes found in graph. Ensure T-box information is present "
            "(e.g. include_t_box=True in get_knowledge_graph)."
        )
    elif len(roots) == 1:
        if (
            wf_name is None
            or wf_name in roots
            or str(wf_name) in [str(v) for v in roots.values()]
        ):
            return next(iter(roots.keys()))
        raise ValueError(
            f"Unknown workflow {wf_name!r}. Available workflow:"
            + f" {list(roots.keys()) + list(roots.values())!r}"
        )
    else:
        if wf_name is None:
            wfs = sorted([str(r) for r in roots.values()])
            wfs = (
                sorted(roots.keys()) + wfs
                if len(wfs) == len(set(wfs))
                else sorted(roots.keys())
            )
            raise ValueError(
                "Graph contains multiple root workflows. Pass `workflow_name`"
                f" explicitly. Available workflows: {wfs}"
            )
        if c := [str(v) for v in roots.values()].count(str(wf_name)):
            if c > 1:
                raise ValueError(
                    f"Ambiguous workflow name {wf_name!r}. It matches {c} workflows."
                    f" Available workflows: {list(roots.keys()) + list(roots.values())!r}"
                )
            return next(k for k, v in roots.items() if str(v) == str(wf_name))
        elif wf_name in roots:
            return cast(URIRef, wf_name)
        else:
            raise ValueError(
                f"Unknown workflow {wf_name!r}. Available workflows:"
                + f" {list(roots.keys()) + list(roots.values())!r}"
            )


def _extract_workflow(
    workflow_graph: nx.DiGraph, workflow_name: str | URIRef
) -> nx.DiGraph:
    for node in nx.weakly_connected_components(workflow_graph):
        if workflow_name in node:
            return workflow_graph.subgraph(node)
    # In principle this error cannot be raised because it is already checked
    # in _ensure_workflow_name, but we keep it for safety.
    raise ValueError(f"Workflow {workflow_name!r} not found in the graph.")


def _rename_workflow(
    workflow_graph: nx.DiGraph,
    uri_to_node: dict[URIRef, Node],
    uri_to_node_and_io: dict[URIRef, Node | IO],
    graph: Graph,
) -> nx.DiGraph:
    return nx.relabel_nodes(workflow_graph, uri_to_node_and_io)


def _kg2digraph(graph: Graph, workflow_name: str | URIRef | None = None) -> nx.DiGraph:
    uri_to_node = _uri_to_node_names(graph)
    all_workflow_graph = _build_workflow_graph(graph)
    workflow_name = _ensure_workflow_name(uri_to_node, workflow_name)
    workflow_graph = _extract_workflow(all_workflow_graph, workflow_name)
    _append_metadata_to_graph(graph, workflow_graph)
    uri_to_node_and_io = _uri_to_node_and_io_names(graph, uri_to_node, workflow_graph)
    renamed_workflow_graph = _rename_workflow(
        workflow_graph, uri_to_node, uri_to_node_and_io, graph
    )
    _reorganize_workflow_graph(renamed_workflow_graph)
    _extract_constant_values_from_kg(graph, renamed_workflow_graph, uri_to_node_and_io)
    _reconstruct_constant_nodes(renamed_workflow_graph)
    return renamed_workflow_graph


def kg2recipe(
    graph: Graph, workflow_name: str | URIRef | None = None
) -> fr.schemas.WorkflowRecipe:
    """
    Translate a Semantikon knowledge graph workflow back to flowrep ``WorkflowRecipe``.

    Args:
        graph (Graph): RDF graph generated by ``semantikon.get_knowledge_graph``.
        workflow_name (str | URIRef | None): Optional root workflow identifier. Can be:
            - String: Local identifier label of the workflow
            - URIRef: Direct URIRef to the workflow
            - None: Required only when the graph contains a single root workflow.

    Returns:
        fr.schemas.WorkflowRecipe: Flowrep workflow recipe.
    """
    selected_workflow = _kg2digraph(graph, workflow_name=workflow_name)
    return _networkx_to_flowrep(selected_workflow)
