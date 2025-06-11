import ast
import builtins
import copy
import inspect
from collections import deque
from functools import cached_property, update_wrapper
from hashlib import sha256
from typing import Any, Callable, Generic, Iterable, TypeVar, cast

import networkx as nx
from networkx.algorithms.dag import topological_sort

from semantikon.converter import (
    get_return_expressions,
    parse_input_args,
    parse_output_args,
)

F = TypeVar("F", bound=Callable[..., object])


class FunctionWithWorkflow(Generic[F]):
    def __init__(self, func: F, workflow, run) -> None:
        self.func = func
        self._semantikon_workflow: dict[str, object] = workflow
        self.run = run
        update_wrapper(self, func)  # Copies __name__, __doc__, etc.

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.func, item)


def ast_from_dict(d):
    """Recursively convert a dict to an ast.AST node"""
    if isinstance(d, dict):
        node_type = getattr(ast, d["_type"])
        fields = {k: ast_from_dict(v) for k, v in d.items() if k != "_type"}
        return node_type(**fields)
    elif isinstance(d, list):
        return [ast_from_dict(x) for x in d]
    else:
        return d


def _extract_variables_from_ast_body(body: dict) -> tuple[set, set]:
    """
    Extracts assigned and used variables from the AST body.

    Args:
        body (dict): The body of the AST function.

    Returns:
        tuple: A tuple containing two sets:
            - assigned_vars: Set of variable names assigned in the body.
            - used_vars: Set of variable names used in the body.
    """
    assigned_vars = set()
    used_vars = set()

    for node in body.get("body", []):
        if node["_type"] == "Assign":
            # Handle left-hand side (targets)
            for target in node["targets"]:
                if target["_type"] == "Name":
                    assigned_vars.add(target["id"])
                elif target["_type"] == "Tuple":
                    for elt in target["elts"]:
                        if elt["_type"] == "Name":
                            assigned_vars.add(elt["id"])

            # Handle right-hand side (value)
            if node["value"]["_type"] == "Call":
                for arg in node["value"]["args"]:
                    if arg["_type"] == "Name":
                        used_vars.add(arg["id"])

    for key in ["test", "or_else", "iter"]:
        if key in body and body[key]["_type"] == "Call":
            for arg in body[key]["args"]:
                if arg["_type"] == "Name":
                    used_vars.add(arg["id"])
    return assigned_vars, used_vars


def _function_to_ast_dict(node):
    if isinstance(node, ast.AST):
        result = {"_type": type(node).__name__}
        for field, value in ast.iter_fields(node):
            result[field] = _function_to_ast_dict(value)
        return result
    elif isinstance(node, list):
        return [_function_to_ast_dict(item) for item in node]
    else:
        return node


def _hash_function(func):
    return f"{func.__name__}_{sha256(inspect.getsource(func).encode()).hexdigest()}"


class InjectedLoop:
    def __init__(self, semantikon_workflow):
        self._semantikon_workflow = semantikon_workflow


class FunctionDictFlowAnalyzer:
    def __init__(self, ast_dict, scope):
        self.graph = nx.DiGraph()
        self.scope = scope  # mapping from function names to objects
        self.function_defs = {}
        self._var_index = {}
        self.ast_dict = ast_dict
        self._call_counter = {}
        self._control_flow_list = []

    def analyze(self):
        for arg in self.ast_dict.get("args", {}).get("args", []):
            if arg["_type"] == "arg":
                self._add_output_edge("input", arg["arg"])
        for node in self.ast_dict.get("body", []):
            self._visit_node(node)
        return self.graph, self.function_defs

    def _visit_node(self, node, control_flow: str | None = None):
        if node["_type"] == "Assign":
            self._handle_assign(node, control_flow=control_flow)
        elif node["_type"] == "Expr":
            self._handle_expr(node, control_flow=control_flow)
        elif node["_type"] == "While":
            self._handle_while(node, control_flow=control_flow)
        elif node["_type"] == "For":
            self._handle_for(node, control_flow=control_flow)
        elif node["_type"] == "Return":
            self._handle_return(node, control_flow=control_flow)

    def _handle_return(self, node, control_flow: str | None = None):
        if not node["value"]:
            return
        if node["value"]["_type"] == "Tuple":
            for idx, elt in enumerate(node["value"]["elts"]):
                if elt["_type"] != "Name":
                    raise NotImplementedError("Only variable returns supported")
                self._add_input_edge(elt, "output", input_index=idx)
        elif node["value"]["_type"] == "Name":
            self._add_input_edge(node["value"], "output")

    def _handle_while(self, node, control_flow: str | None = None):
        if node["test"]["_type"] != "Call":
            raise NotImplementedError("Only function calls allowed in while test")
        if control_flow is None:
            control_flow = ""
        else:
            control_flow = f"{control_flow.split('-')[0]}/"
        counter = 0
        while True:
            if f"{control_flow}While_{counter}" not in self._control_flow_list:
                self._control_flow_list.append(f"{control_flow}While_{counter}")
                break
            counter += 1
        control_flow = f"{control_flow}While_{counter}"
        self._parse_function_call(node["test"], control_flow=f"{control_flow}-test")
        for node in node["body"]:
            self._visit_node(node, control_flow=f"{control_flow}-body")

    def _handle_for(self, node, control_flow: str | None = None):
        if node["iter"]["_type"] != "Call":
            raise NotImplementedError("Only function calls allowed in while test")

    def _handle_expr(self, node, control_flow: str | None = None):
        value = node["value"]
        return self._parse_function_call(value, control_flow=control_flow)

    def _parse_function_call(self, value, control_flow: str | None = None):
        if value["_type"] != "Call":
            raise NotImplementedError("Only function calls allowed on RHS")

        func_node = value["func"]
        if func_node["_type"] != "Name":
            raise NotImplementedError("Only simple functions allowed")

        func_name = func_node["id"]
        unique_func_name = self._get_unique_func_name(func_name)

        if func_name not in self.scope:
            raise ValueError(f"Function {func_name} not found in scope")

        self.function_defs[unique_func_name] = {"function": self.scope[func_name]}
        if control_flow is not None:
            self.function_defs[unique_func_name]["control_flow"] = control_flow

        # Parse inputs (positional + keyword)
        for i, arg in enumerate(value.get("args", [])):
            self._add_input_edge(
                arg, unique_func_name, input_index=i, control_flow=control_flow
            )
        for kw in value.get("keywords", []):
            self._add_input_edge(
                kw["value"],
                unique_func_name,
                input_name=kw["arg"],
                control_flow=control_flow,
            )
        return unique_func_name

    def _handle_assign(self, node, control_flow: str | None = None):
        unique_func_name = self._handle_expr(node, control_flow=control_flow)
        # Parse outputs
        self._parse_outputs(
            node["targets"], unique_func_name, control_flow=control_flow
        )

    def _parse_outputs(
        self, targets, unique_func_name, control_flow: str | None = None
    ):
        if len(targets) == 1 and targets[0]["_type"] == "Tuple":
            for idx, elt in enumerate(targets[0]["elts"]):
                self._add_output_edge(
                    unique_func_name,
                    elt["id"],
                    output_index=idx,
                    control_flow=control_flow,
                )
        else:
            for target in targets:
                self._add_output_edge(
                    unique_func_name, target["id"], control_flow=control_flow
                )

    def _add_output_edge(
        self, source, target, control_flow: str | None = None, **kwargs
    ):
        self._var_index[target] = self._var_index.get(target, -1) + 1
        versioned = f"{target}_{self._var_index[target]}"
        if control_flow is not None:
            kwargs["control_flow"] = control_flow
        self.graph.add_edge(source, versioned, type="output", **kwargs)

    def _add_input_edge(
        self, source, target, control_flow: str | None = None, **kwargs
    ):
        if source["_type"] != "Name":
            raise NotImplementedError(f"Only variable inputs supported, got: {source}")
        var_name = source["id"]
        if control_flow is not None:
            kwargs["control_flow"] = control_flow
        if var_name not in self._var_index:
            raise ValueError(f"Variable {var_name} not found in scope")
        idx = self._var_index[var_name]
        versioned = f"{var_name}_{idx}"
        self.graph.add_edge(versioned, target, type="input", **kwargs)

    def _get_unique_func_name(self, base_name):
        i = self._call_counter.get(base_name, 0)
        self._call_counter[base_name] = i + 1
        return f"{base_name}_{i}"


def _get_variables_from_subgraph(graph: nx.DiGraph, io_: str) -> set[str]:
    """
    Get variables from a subgraph based on the type of I/O and control flow.

    Args:
        graph (nx.DiGraph): The directed graph representing the function.
        io_ (str): The type of I/O to filter by ("input" or "output").
        control_flow (list, str): A list of control flow types to filter by.

    Returns:
        set[str]: A set of variable names that match the specified I/O type and
            control flow.
    """
    if io_ == "input":
        edge_ind = 0
    elif io_ == "output":
        edge_ind = 1
    else:
        raise ValueError(f"Invalid I/O type: {io_}. Expected 'input' or 'output'.")
    return set(
        [edge[edge_ind] for edge in graph.edges.data() if edge[2]["type"] == io_]
    )


def _get_parent_graph(graph: nx.DiGraph, control_flow: str) -> nx.DiGraph:
    cf_list = [""]
    for cf in control_flow.split("/")[:-1]:
        cf_list.append("/".join([cf_list[-1], cf]))
    return nx.DiGraph(
        [
            edge
            for cf in cf_list
            for edge in graph.edges.data()
            if edge[2].get("control_flow", "").split("-")[0] == cf
        ]
    )


def _detect_io_variables_from_control_flow(
    graph: nx.DiGraph, subgraph: nx.DiGraph, control_flow
) -> dict[str, list]:
    """
    Detect input and output variables from a graph based on control flow.

    Args:
        graph (nx.DiGraph): The directed graph representing the function.
        control_flow (str | list): The type of control flow to filter by.

    Returns:
        dict[str, set]: A dictionary with keys "input" and "output", each
            containing a set

    Take a look at the unit tests for examples of how to use this function.
    """
    parent_graph = _get_parent_graph(graph, control_flow)
    var_inp_1 = _get_variables_from_subgraph(graph=subgraph, io_="input")
    var_inp_2 = _get_variables_from_subgraph(graph=parent_graph, io_="output")
    var_out_1 = _get_variables_from_subgraph(graph=parent_graph, io_="input")
    var_out_2 = _get_variables_from_subgraph(graph=subgraph, io_="output")
    return {
        "inputs": list(var_inp_1.intersection(var_inp_2)),
        "outputs": list(var_out_1.intersection(var_out_2)),
    }


def _extract_control_flows(graph: nx.DiGraph) -> list[str]:
    return list(
        set(
            [
                edge[2].get("control_flow", "").split("-")[0]
                for edge in graph.edges.data()
            ]
        )
    )


def _get_subgraphs(graph: nx.DiGraph) -> dict[str, nx.DiGraph]:
    return {
        control_flow: nx.DiGraph(
            [
                edge
                for edge in graph.edges.data()
                if edge[2].get("control_flow", "").split("-")[0] == control_flow
            ]
        )
        for control_flow in _extract_control_flows(graph)
    }


def _extract_functions_from_graph(graph: nx.DiGraph) -> set:
    function_names = []
    for edge in graph.edges.data():
        if edge[2]["type"] == "output" and edge[0] != "input":
            function_names.append(edge[0])
        elif edge[2]["type"] == "input" and edge[1] != "output":
            function_names.append(edge[1])
    return set(function_names)


def get_ast_dict(func: Callable) -> dict:
    """Get the AST dictionary representation of a function."""
    source_code = inspect.getsource(func)
    tree = ast.parse(source_code)
    return _function_to_ast_dict(tree)


def analyze_function(func):
    """Extracts the variable flow graph from a function"""
    ast_dict = get_ast_dict(func)
    scope = inspect.getmodule(func).__dict__ | vars(builtins)
    analyzer = FunctionDictFlowAnalyzer(ast_dict["body"][0], scope)
    return analyzer.analyze()


def _get_workflow_outputs(func):
    var_output = get_return_expressions(func)
    if isinstance(var_output, str):
        var_output = [var_output]
    data_output = parse_output_args(func)
    if isinstance(data_output, dict):
        data_output = [data_output]
    if len(var_output) > 1 and len(data_output) == 1:
        assert len(data_output[0]) == 0
        return {var: {} for var in var_output}
    return dict(zip(var_output, data_output))


def _get_node_outputs(func: Callable, counts: int) -> dict[str, dict]:
    output_hints = parse_output_args(func, separate_tuple=counts > 1)
    output_vars = get_return_expressions(func)
    if output_vars is None or len(output_vars) == 0:
        return {}
    if counts == 1:
        if isinstance(output_vars, str):
            return {output_vars: cast(dict, output_hints)}
        else:
            return {"output": cast(dict, output_hints)}
    assert isinstance(output_vars, tuple) and len(output_vars) == counts
    assert len(output_vars) == counts
    if output_hints == {}:
        return {key: {} for key in output_vars}
    else:
        assert len(output_hints) == counts
        return {key: hint for key, hint in zip(output_vars, output_hints)}


def _get_output_counts(graph: nx.DiGraph) -> dict:
    """
    Get the number of outputs for each node in the graph.

    Args:
        graph (nx.DiGraph): The directed graph representing the function.

    Returns:
        dict: A dictionary mapping node names to the number of outputs.
    """
    f_dict: dict = {}
    for edge in graph.edges.data():
        if edge[2]["type"] != "output":
            continue
        f_dict[edge[0]] = f_dict.get(edge[0], 0) + 1
    if "input" in f_dict:
        del f_dict["input"]
    return f_dict


def _get_nodes(data: dict[str, dict], output_counts: dict[str, int], control_flow: None | str=None) -> dict[str, dict]:
    result = {}
    for node, function in data.items():
        func = function["function"]
        if hasattr(func, "_semantikon_workflow"):
            data_dict = func._semantikon_workflow.copy()
            result[node] = data_dict
            result[node]["label"] = node
        else:
            result[node] = _to_node_dict_entry(
                func,
                parse_input_args(func),
                _get_node_outputs(func, output_counts.get(node, 1)),
            )
        if hasattr(func, "_semantikon_metadata"):
            result[node].update(func._semantikon_metadata)
    return result


def _remove_index(s):
    return "_".join(s.split("_")[:-1])


def _get_sorted_edges(graph: nx.DiGraph) -> list:
    """
    Sort the edges of the graph based on the topological order of the nodes.

    Args:
        graph (nx.DiGraph): The directed graph representing the function.

    Returns:
        list: A sorted list of edges in the graph.

    Example:

    >>> graph.add_edges_from([('A', 'B'), ('B', 'D'), ('A', 'C'), ('C', 'D')])
    >>> sorted_edges = _get_sorted_edges(graph)
    >>> print(sorted_edges)

    Output:

    >>> [('A', 'B', {}), ('A', 'C', {}), ('B', 'D', {}), ('C', 'D', {})]
    """
    topo_order = list(topological_sort(graph))
    node_order = {node: i for i, node in enumerate(topo_order)}
    return sorted(graph.edges.data(), key=lambda edge: node_order[edge[0]])


def _remove_and_reconnect_nodes(
    G: nx.DiGraph, nodes_to_remove: list[str]
) -> nx.DiGraph:
    for node in set(nodes_to_remove):
        preds = list(G.predecessors(node))
        succs = list(G.successors(node))
        for u in preds:
            for v in succs:
                G.add_edge(u, v)
        G.remove_node(node)
    return G


def _get_edges(
    graph: nx.DiGraph, nodes: dict[str, dict]
) -> list[tuple[str, str]]:
    io_dict = {
        key: {
            "input": list(data["inputs"].keys()),
            "output": list(data["outputs"].keys()),
        }
        for key, data in nodes.items()
    }
    edges = []
    nodes_to_remove = []
    for edge in graph.edges.data():
        if edge[0] == "input":
            edges.append([edge[0] + "s." + edge[1].split("_")[0], edge[1]])
            nodes_to_remove.append(edge[1])
        elif edge[1] == "output":
            edges.append([edge[0], edge[1] + "s." + edge[0].split("_")[0]])
            nodes_to_remove.append(edge[0])
        elif edge[2]["type"] == "input":
            if "input_name" in edge[2]:
                tag = edge[2]["input_name"]
            elif "input_index" in edge[2]:
                tag = io_dict[edge[1]]["input"][edge[2]["input_index"]]
            else:
                raise ValueError
            edges.append([edge[0], edge[1] + ".inputs." + tag])
            nodes_to_remove.append(edge[0])
        elif edge[2]["type"] == "output":
            if "output_index" in edge[2]:
                tag = io_dict[edge[0]]["output"][edge[2]["output_index"]]
            else:
                tag = io_dict[edge[0]]["output"][0]
            edges.append([edge[0] + ".outputs." + tag, edge[1]])
            nodes_to_remove.append(edge[1])
    new_graph = _remove_and_reconnect_nodes(nx.DiGraph(edges), nodes_to_remove)
    return list(new_graph.edges)


def _dtype_to_str(dtype):
    return dtype.__name__


def _to_ape(data, func):
    data["taxonomyOperations"] = [data.pop("uri", func.__name__)]
    data["id"] = data["label"] + "_" + _hash_function(func)
    for io_ in ["inputs", "outputs"]:
        d = []
        for v in data[io_].values():
            if "uri" in v:
                d.append({"Type": str(v["uri"]), "Format": _dtype_to_str(v["dtype"])})
            else:
                d.append({"Type": _dtype_to_str(v["dtype"])})
        data[io_] = d
    return data


def get_node_dict(func, data_format="semantikon"):
    """
    Get a dictionary representation of the function node.

    Args:
        func (Callable): The function to be analyzed.
        data_format (str): The format of the output. Options are "semantikon" and
            "ape".

    Returns:
        (dict) A dictionary representation of the function node.
    """
    data = {
        "inputs": parse_input_args(func),
        "outputs": _get_workflow_outputs(func),
        "label": func.__name__,
    }
    if hasattr(func, "_semantikon_metadata"):
        data.update(func._semantikon_metadata)
    if data_format.lower() == "ape":
        return _to_ape(data, func)
    return data


def separate_types(
    data: dict[str, Any], class_dict: dict[str, type] | None = None
) -> tuple[dict[str, Any], dict[str, type]]:
    """
    Separate types from the data dictionary and store them in a class dictionary.
    The types inside the data dictionary will be replaced by their name (which
    would for example make it easier to hash it).

    Args:
        data (dict[str, Any]): The data dictionary containing nodes and types.
        class_dict (dict[str, type], optional): A dictionary to store types. It
            is mainly used due to the recursivity of this function. Defaults to
            None.

    Returns:
        tuple: A tuple containing the modified data dictionary and the
            class dictionary.
    """
    data = copy.deepcopy(data)
    if class_dict is None:
        class_dict = {}
    if "nodes" in data:
        for key, node in data["nodes"].items():
            child_node, child_class_dict = separate_types(node, class_dict)
            class_dict.update(child_class_dict)
            data["nodes"][key] = child_node
    for io_ in ["inputs", "outputs"]:
        for key, content in data[io_].items():
            if "dtype" in content and isinstance(content["dtype"], type):
                class_dict[content["dtype"].__name__] = content["dtype"]
                data[io_][key]["dtype"] = content["dtype"].__name__
    return data, class_dict


def separate_functions(
    data: dict[str, Any], function_dict: dict[str, Callable] | None = None
) -> tuple[dict[str, Any], dict[str, Callable]]:
    """
    Separate functions from the data dictionary and store them in a function
    dictionary. The functions inside the data dictionary will be replaced by
    their name (which would for example make it easier to hash it)

    Args:
        data (dict[str, Any]): The data dictionary containing nodes and
            functions.
        function_dict (dict[str, Callable], optional): A dictionary to store
            functions. It is mainly used due to the recursivity of this
            function. Defaults to None.

    Returns:
        tuple: A tuple containing the modified data dictionary and the
            function dictionary.
    """
    data = copy.deepcopy(data)
    if function_dict is None:
        function_dict = {}
    if "nodes" in data:
        for key, node in data["nodes"].items():
            child_node, child_function_dict = separate_functions(node, function_dict)
            function_dict.update(child_function_dict)
            data["nodes"][key] = child_node
    elif "function" in data and not isinstance(data["function"], str):
        function_dict[data["function"].__name__] = data["function"]
        data["function"] = data["function"].__name__
    if "test" in data and not isinstance(data["test"]["function"], str):
        function_dict[data["test"]["function"].__name__] = data["test"]["function"]
        data["test"]["function"] = data["test"]["function"].__name__
    return data, function_dict


def _to_node_dict_entry(
    function: Callable, inputs: dict[str, dict], outputs: dict[str, dict]
) -> dict:
    return {"function": function, "inputs": inputs, "outputs": outputs}


def _to_workflow_dict_entry(
    inputs: dict[str, dict],
    outputs: dict[str, dict],
    nodes: dict[str, dict],
    edges: list[tuple[str, str]],
    label: str,
    **kwargs,
) -> dict[str, object]:
    assert all("inputs" in v for v in nodes.values())
    assert all("outputs" in v for v in nodes.values())
    assert all(
        "function" in v or ("nodes" in v and "edges" in v) for v in nodes.values()
    )
    return {
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes,
        "edges": edges,
        "label": label,
    } | kwargs


def get_workflow_dict(func: Callable) -> dict[str, object]:
    """
    Get a dictionary representation of the workflow for a given function.

    Args:
        func (Callable): The function to be analyzed.

    Returns:
        dict: A dictionary representation of the workflow, including inputs,
            outputs, nodes, edges, and label.
    """
    graph, f_dict = analyze_function(func)
    nodes = _get_nodes(f_dict, _get_output_counts(graph))
    return _to_workflow_dict_entry(
        inputs=parse_input_args(func),
        outputs=_get_workflow_outputs(func),
        nodes=nodes,
        edges=_get_edges(graph, nodes),
        label=func.__name__,
    )


def _get_missing_edges(edge_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Insert processes into the data edges. Take the following workflow:

    >>> y = f(x=x)
    >>> z = g(y=y)

    The data flow is

    - f.inputs.x -> f.outputs.y
    - f.outputs.y -> g.inputs.y
    - g.inputs.y -> g.outputs.z

    `_get_missing_edges` adds the processes:

    - f.inputs.x -> f
    - f -> f.outputs.y
    - f.outputs.y -> g.inputs.y
    - g.inputs.y -> g
    - g -> g.outputs.z
    """
    extra_edges = []
    for edge in edge_list:
        for tag in edge:
            if len(tag.split(".")) < 3:
                continue
            if tag.split(".")[1] == "inputs":
                new_edge = (tag, tag.split(".")[0])
            elif tag.split(".")[1] == "outputs":
                new_edge = (tag.split(".")[0], tag)
            if new_edge not in extra_edges:
                extra_edges.append(new_edge)
    return extra_edges


class _Workflow:
    def __init__(self, workflow_dict: dict[str, Any]):
        self._workflow = workflow_dict

    @cached_property
    def _all_edges(self) -> list[tuple[str, str]]:
        edges = cast(dict[str, list], self._workflow)["edges"]
        return edges + _get_missing_edges(edges)

    @cached_property
    def _graph(self) -> nx.DiGraph:
        return nx.DiGraph(self._all_edges)

    @cached_property
    def _execution_list(self) -> list[list[str]]:
        return find_parallel_execution_levels(self._graph)

    def _sanitize_input(self, *args, **kwargs) -> dict[str, Any]:
        keys = list(self._workflow["inputs"].keys())
        for ii, arg in enumerate(args):
            if keys[ii] in kwargs:
                raise TypeError(
                    f"{self._workflow['label']}() got multiple values for"
                    " argument '{keys[ii]}'"
                )
            kwargs[keys[ii]] = arg
        return kwargs

    def _set_inputs(self, *args, **kwargs):
        kwargs = self._sanitize_input(*args, **kwargs)
        for key, value in kwargs.items():
            if key not in self._workflow["inputs"]:
                raise TypeError(f"Unexpected keyword argument '{key}'")
            self._workflow["inputs"][key]["value"] = value

    def _get_value_from_data(self, node: dict[str, Any]) -> Any:
        if "value" not in node:
            node["value"] = node["default"]
        return node["value"]

    def _get_value_from_global(self, path: str) -> Any:
        io, var = path.split(".")
        return self._get_value_from_data(self._workflow[io][var])

    def _get_value_from_node(self, path: str) -> Any:
        node, io, var = path.split(".")
        return self._get_value_from_data(self._workflow["nodes"][node][io][var])

    def _set_value_from_global(self, path, value):
        io, var = path.split(".")
        self._workflow[io][var]["value"] = value

    def _set_value_from_node(self, path, value):
        node, io, var = path.split(".")
        try:
            self._workflow["nodes"][node][io][var]["value"] = value
        except KeyError:
            raise KeyError(f"{path} not found in {node}")

    def _execute_node(self, function: str) -> Any:
        node = self._workflow["nodes"][function]
        input_data = {}
        try:
            for key, content in node["inputs"].items():
                if "value" not in content:
                    content["value"] = content["default"]
                input_data[key] = content["value"]
        except KeyError:
            raise KeyError(f"value not defined for {function}")
        if "function" not in node:
            workflow = _Workflow(node)
            outputs = [
                d["value"] for d in workflow.run(**input_data)["outputs"].values()
            ]
            if len(outputs) == 1:
                outputs = outputs[0]
        else:
            outputs = node["function"](**input_data)
        return outputs

    def _set_value(self, tag, value):
        if len(tag.split(".")) == 2 and tag.split(".")[0] in ("inputs", "outputs"):
            self._set_value_from_global(tag, value)
        elif len(tag.split(".")) == 3 and tag.split(".")[1] in ("inputs", "outputs"):
            self._set_value_from_node(tag, value)
        elif "." in tag:
            raise ValueError(f"{tag} not recognized")

    def _get_value(self, tag: str):
        if len(tag.split(".")) == 2 and tag.split(".")[0] in ("inputs", "outputs"):
            return self._get_value_from_global(tag)
        elif len(tag.split(".")) == 3 and tag.split(".")[1] in ("inputs", "outputs"):
            return self._get_value_from_node(tag)
        elif "." not in tag:
            return self._execute_node(tag)
        else:
            raise ValueError(f"{tag} not recognized")

    def run(self, *args, **kwargs) -> dict[str, Any]:
        self._set_inputs(*args, **kwargs)
        for current_list in self._execution_list:
            for item in current_list:
                values = self._get_value(item)
                nodes = self._graph.edges(item)
                if "." not in item and len(nodes) > 1:
                    for value, node in zip(values, nodes):
                        self._set_value(node[1], value)
                else:
                    for node in nodes:
                        self._set_value(node[1], values)
        return self._workflow


def find_parallel_execution_levels(G: nx.DiGraph) -> list[list[str]]:
    """
    Find levels of parallel execution in a directed acyclic graph (DAG).

    Args:
        G (nx.DiGraph): The directed graph representing the function.

    Returns:
        list[list[str]]: A list of lists, where each inner list contains nodes
            that can be executed in parallel.

    Comment:
        This function only gives you a list of nodes that can be executed in
        parallel, but does not tell you which processes can be executed in
        case there is a process that takes longer at a higher level.
    """
    in_degree = dict(cast(Iterable[tuple[Any, int]], G.in_degree()))
    queue = deque([node for node in G.nodes if in_degree[node] == 0])
    levels = []

    while queue:
        current_level = list(queue)
        if "input" not in current_level and "output" not in current_level:
            levels.append(current_level)

        next_queue: deque = deque()
        for node in current_level:
            for neighbor in G.successors(node):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_queue.append(neighbor)

        queue = next_queue

    return levels


def workflow(func: Callable) -> FunctionWithWorkflow:
    """
    Decorator to convert a function into a workflow with metadata.

    Args:
        func (Callable): The function to be converted into a workflow.

    Returns:
        FunctionWithWorkflow: A callable object that includes the original function

    Example:

    >>> def operation(x: float, y: float) -> tuple[float, float]:
    >>>     return x + y, x - y
    >>>
    >>>
    >>> def add(x: float = 2.0, y: float = 1) -> float:
    >>>     return x + y
    >>>
    >>>
    >>> def multiply(x: float, y: float = 5) -> float:
    >>>     return x * y
    >>>
    >>>
    >>> @workflow
    >>> def example_macro(a=10, b=20):
    >>>     c, d = operation(a, b)
    >>>     e = add(c, y=d)
    >>>     f = multiply(e)
    >>>     return f
    >>>
    >>>
    >>> @workflow
    >>> def example_workflow(a=10, b=20):
    >>>     y = example_macro(a, b)
    >>>     z = add(y, b)
    >>>     return z

    This example defines a workflow `example_macro`, that includes `operation`,
    `add`, and `multiply`, which is nested inside another workflow
    `example_workflow`. Both workflows can be executed using their `run` method,
    which returns the dictionary representation of the workflow with all the
    intermediate steps and outputs.
    """
    workflow_dict = get_workflow_dict(func)
    w = _Workflow(workflow_dict)
    func_with_metadata = FunctionWithWorkflow(func, workflow_dict, w.run)
    return func_with_metadata
