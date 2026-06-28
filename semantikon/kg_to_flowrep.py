from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any, cast

import flowrep as fr
import networkx as nx
from pyiron_snippets import retrieve
from rdflib import OWL, RDF, RDFS, Graph, Literal, URIRef
from rdflib.namespace import SH
from rdflib.term import IdentifiedNode

from semantikon.flowrep_dict import dict_to_nodedata
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

    data: dict[str, Any] = {"qualname": qualname.toPython()}

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
            data["docstring"] = docstring.toPython()

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
            data["hash"] = hash_value.toPython()

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
            data["module"] = module.toPython()

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

            # First collect all direct children
            direct_children = {}
            for child_label, child_node in G.nodes.items():
                if (
                    child_node.get("step") == "node"
                    and child_node.get("parent") == node_name
                ):
                    # Extract child short label correctly by removing parent prefix
                    if child_label.startswith(node_name + "-"):
                        child_short_label = child_label[len(node_name) + 1 :]
                    else:
                        child_short_label = child_label.split("-", 1)[1]
                    direct_children[child_short_label] = child_label
                    nodes[child_short_label] = _process_node(child_label)

            def _find_child_for_io(io_node_name: str) -> str | None:
                """Find the child node (short label) that owns this IO node."""
                for child_label in direct_children.values():
                    if io_node_name.startswith(child_label + "-"):
                        # Extract child short label correctly
                        if child_label.startswith(node_name + "-"):
                            child_short_label = child_label[len(node_name) + 1 :]
                        else:
                            child_short_label = child_label.split("-", 1)[1]
                        return child_short_label
                return None

            def _is_direct_io(node_id: str) -> bool:
                """Check if this is a direct IO of the workflow."""
                if not node_id.startswith(node_name + "-"):
                    return False
                rest = node_id[len(node_name) + 1 :]
                parts = rest.split("-")
                # Direct IO is like "inputs-arg" or "outputs-arg" (2 parts)
                return len(parts) == 2

            def _is_child_io(node_id: str) -> bool:
                """Check if this is an IO of a direct child (and only direct child)."""
                for child_label in direct_children.values():
                    if node_id.startswith(child_label + "-"):
                        # Make sure it's not a grandchild IO
                        rest = node_id[len(child_label) + 1 :]
                        parts = rest.split("-")
                        # Direct child IO is like "inputs-arg" or "outputs-arg" (2 parts)
                        if len(parts) == 2:
                            node = G.nodes.get(node_id)
                            if node and node.get("step") in ["inputs", "outputs"]:
                                return True
                return False

            for u, v in G.edges:
                u_data = G.nodes[u]
                v_data = G.nodes[v]
                u_step = u_data.get("step")
                v_step = v_data.get("step")

                u_is_direct_io = _is_direct_io(u)
                v_is_direct_io = _is_direct_io(v)
                u_is_child_io = _is_child_io(u)
                v_is_child_io = _is_child_io(v)

                if u_step == "inputs" and v_step == "inputs":
                    if u_is_direct_io and v_is_child_io:
                        u_port = u_data["arg"]
                        v_child = _find_child_for_io(v)
                        if v_child is not None:
                            v_port = v_data["arg"]
                            edges.append(
                                (f"inputs.{u_port}", f"{v_child}.inputs.{v_port}")
                            )
                elif u_step == "outputs" and v_step == "inputs":
                    if u_is_child_io and v_is_child_io:
                        u_child = _find_child_for_io(u)
                        v_child = _find_child_for_io(v)
                        if u_child is not None and v_child is not None:
                            u_port = u_data["arg"]
                            v_port = v_data["arg"]
                            edges.append(
                                (
                                    f"{u_child}.outputs.{u_port}",
                                    f"{v_child}.inputs.{v_port}",
                                )
                            )
                elif u_step == "inputs" and v_step == "outputs":
                    if u_is_direct_io and v_is_direct_io:
                        u_port = u_data["arg"]
                        v_port = v_data["arg"]
                        edges.append((f"inputs.{u_port}", f"outputs.{v_port}"))
                elif u_step == "outputs" and v_step == "outputs" and u != v:
                    if u_is_child_io and v_is_direct_io:
                        u_child = _find_child_for_io(u)
                        if u_child is not None:
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
        return local_identifier.toPython()
    label = graph.value(node, RDFS.label)
    if label is not None:
        return label.toPython()
    return str(node)


def _label(graph: Graph, node: URIRef) -> str:
    label = graph.value(node, RDFS.label)
    if label is not None:
        return label.toPython()
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


def _reorganize_output_edges(
    graph: nx.DiGraph, node: URIRef, position: dict[URIRef, int]
):
    io_dict: dict[URIRef, URIRef] = {}
    for n in graph.predecessors(node):
        pred = list(graph.predecessors(n))
        assert len(pred) == 1 and pred[0] not in io_dict
        io_dict[pred[0]] = n
    keys = sorted(io_dict.keys(), key=position.get)[::-1]
    nodes = [io_dict[k] for k in keys]
    for n in nodes[:-1]:
        graph.remove_edge(n, node)
    graph.add_edges_from(zip(nodes[:-1], nodes[1:]))


def _reorganize_input_edges(
    graph: nx.DiGraph, node: URIRef, position: dict[URIRef, int]
):
    io_dict: dict[URIRef, URIRef] = {}
    for n in graph.successors(node):
        succ = list(graph.successors(n))
        assert len(succ) == 1 and succ[0] not in io_dict
        io_dict[succ[0]] = n
    node_keys = sorted(io_dict.keys(), key=position.get)
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


def _add_io_nodes(
    graph: Graph,
    workflow_graph: nx.DiGraph,
    function_dict: dict[URIRef, dict[str, Any]],
    node_function_dict: dict[URIRef, URIRef],
    io_type: str,
):
    if io_type == "input":
        io_query = graph.query(
            _get_connection_query(SNS.workflow_node, SNS.has_part, SNS.input_assignment)
        )
    else:
        io_query = graph.query(
            _get_connection_query(
                SNS.workflow_node, SNS.has_part, SNS.output_assignment
            )
        )
    for node, io_node in io_query:
        if io_type == "input":
            workflow_graph.add_edge(io_node, node)
        else:
            workflow_graph.add_edge(node, io_node)

        function_data = function_dict.get(node_function_dict.get(node))
        if function_data is None:
            continue
        workflow_graph.add_node(node, step="node", function=function_data["data"])

        arg_values = list(graph.objects(io_node, SNS.local_identifier))
        if len(arg_values) != 1:
            raise ValueError(f"Expected one local identifier for {io_node!r}.")
        arg = arg_values[0].toPython()
        for data in function_data[f"{io_type}_args"]:
            if arg == data["arg"]:
                workflow_graph.add_node(io_node, step=f"{io_type}s", **data)
                break
        else:
            raise ValueError(f"Could not match argument {arg!r} for {node!r}.")


def _build_workflow_graph(graph: Graph) -> nx.DiGraph:
    function_nodes = list(graph.subjects(RDF.type, SNS.workflow_function))
    function_dict = {
        f_node: _graph_to_function(graph, f_node) for f_node in function_nodes
    }
    node_function_dict = _node_functions(graph)

    workflow_graph = nx.DiGraph()
    _add_io_nodes(
        graph, workflow_graph, function_dict, node_function_dict, io_type="input"
    )
    _add_io_nodes(
        graph, workflow_graph, function_dict, node_function_dict, io_type="output"
    )

    for out_assignment, data_node in graph.query(
        _get_connection_query(
            SNS.output_assignment, SNS.has_participant, SNS.value_specification
        )
    ):
        workflow_graph.add_edge(out_assignment, data_node)
        workflow_graph.add_node(data_node, step="data")

    for in_assignment, data_node in graph.query(
        _get_connection_query(
            SNS.input_assignment, SNS.has_participant, SNS.value_specification
        )
    ):
        workflow_graph.add_edge(data_node, in_assignment)
        workflow_graph.add_node(data_node, step="data")

    for parent, child in graph.query(
        _get_connection_query(SNS.workflow_node, SNS.has_part, SNS.workflow_node)
    ):
        workflow_graph.add_edge(parent, child)
        if parent not in workflow_graph:
            workflow_graph.add_node(parent, step="node")
        if child not in workflow_graph:
            workflow_graph.add_node(child, step="node")
        workflow_graph.nodes[child]["parent"] = _label(graph, parent)
        workflow_graph.nodes[parent]["type"] = "workflow"
        workflow_graph.nodes[child]["type"] = workflow_graph.nodes[child].get(
            "type", "atomic"
        )

    position = {
        node: i
        for i, node in enumerate(nx.topological_sort(workflow_graph))
        if workflow_graph.nodes[node].get("step") == "node"
    }
    for node, data in tuple(workflow_graph.nodes.data()):
        if data.get("step") != "data":
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
        [
            node
            for node, data in workflow_graph.nodes.data()
            if data.get("step") == "data"
        ]
    )
    workflow_graph.remove_edges_from(
        [
            edge
            for edge in workflow_graph.edges
            if all(workflow_graph.nodes[node].get("step") == "node" for node in edge)
        ]
    )
    mapping = {node: _label(graph, node) for node in workflow_graph.nodes}
    return nx.relabel_nodes(workflow_graph, mapping)


def _workflow_roots(graph: Graph) -> dict[str, URIRef]:
    node_graph = nx.DiGraph()
    workflow_nodes = list(graph.subjects(RDFS.subClassOf, SNS.workflow_node))
    node_graph.add_nodes_from(workflow_nodes)
    node_graph.add_edges_from(
        graph.query(
            _get_connection_query(SNS.workflow_node, SNS.has_part, SNS.workflow_node)
        )
    )
    roots = [node for node in node_graph.nodes if node_graph.in_degree(node) == 0]
    return {_label(graph, root): root for root in roots}


def _split_by_roots(
    graph: Graph, workflow_graph: nx.DiGraph, roots: dict[str, URIRef]
) -> dict[str, nx.DiGraph]:
    subgraphs: dict[str, nx.DiGraph] = {}
    for component in nx.weakly_connected_components(workflow_graph):
        component_nodes = set(component)
        component_workflows = component_nodes & set(roots.keys())
        if len(component_workflows) == 0:
            raise ValueError("Could not assign graph component to a root workflow.")
        if len(component_workflows) > 1:
            raise ValueError(
                "A graph component contains more than one root workflow: "
                f"{sorted(component_workflows)}"
            )
        name = next(iter(component_workflows))
        subgraphs[name] = workflow_graph.subgraph(component).copy()
        subgraphs[name].name = name
    return subgraphs


def _select_workflow(
    graph: Graph,
    roots: dict[str, URIRef],
    workflows: Iterable[str],
    workflow_name: str | None,
) -> str:
    names = sorted(set(workflows))
    if workflow_name is not None:
        if workflow_name in names:
            return workflow_name
        by_identifier: dict[str, list[str]] = {}
        for label, uri in roots.items():
            by_identifier.setdefault(_identifier(graph, uri), []).append(label)
        matches = by_identifier.get(workflow_name, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Workflow identifier {workflow_name!r} is ambiguous. Matches: {matches}"
            )
        if workflow_name not in names:
            raise ValueError(
                f"Unknown workflow {workflow_name!r}. Available workflows: {names}"
            )
    if len(names) != 1:
        raise ValueError(
            "Graph contains multiple root workflows. Pass `workflow_name` explicitly. "
            f"Available workflows: {names}"
        )
    return names[0]


def knowledge_graph_to_flowrep_data(
    graph: Graph, workflow_name: str | None = None
) -> fr.schemas.DagData:
    """
    Translate a Semantikon knowledge graph workflow back to flowrep ``DagData``.

    Args:
        graph (Graph): RDF graph generated by ``semantikon.get_knowledge_graph``.
        workflow_name (str | None): Optional root workflow local identifier. Required
            when the graph contains multiple root workflows.

    Returns:
        fr.schemas.DagData: Flowrep workflow data.
    """
    roots = _workflow_roots(graph)
    if len(roots) == 0:
        raise ValueError(
            "No workflow nodes found in graph. Ensure T-box information is present "
            "(e.g. include_t_box=True in get_knowledge_graph)."
        )
    workflow_graph = _build_workflow_graph(graph)
    workflows = _split_by_roots(graph, workflow_graph, roots)
    selected_name = _select_workflow(
        graph, roots, workflows.keys(), workflow_name=workflow_name
    )
    return serialize_and_networkx_to_data(workflows[selected_name])


def knowledge_graph_to_flowrep_recipe(
    graph: Graph, workflow_name: str | None = None
) -> fr.schemas.WorkflowRecipe:
    """
    Translate a Semantikon knowledge graph workflow back to flowrep ``WorkflowRecipe``.

    Args:
        graph (Graph): RDF graph generated by ``semantikon.get_knowledge_graph``.
        workflow_name (str | None): Optional root workflow local identifier.

    Returns:
        fr.schemas.WorkflowRecipe: Flowrep workflow recipe.
    """
    return knowledge_graph_to_flowrep_data(graph, workflow_name=workflow_name).recipe
