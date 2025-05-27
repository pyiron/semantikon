import ast
import builtins
import copy
import inspect
from collections import deque
from functools import cached_property
from hashlib import sha256

import networkx as nx
from networkx.algorithms.dag import topological_sort

from semantikon.converter import (
    get_return_expressions,
    parse_input_args,
    parse_output_args,
)


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

    if "test" in body:
        if body["test"]["_type"] == "Call":
            for arg in body["test"]["args"]:
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
    def __init__(self, ast_dict, scope, input_vars=None):
        if input_vars is None:
            input_vars = []
        self.graph = nx.DiGraph()
        self.scope = scope  # mapping from function names to objects
        self.function_defs = {}
        self._var_index = {input_var: 0 for input_var in input_vars}
        self.ast_dict = ast_dict
        self._call_counter = {}

    def analyze(self):
        for arg in self.ast_dict.get("args", {}).get("args", []):
            if arg["_type"] == "arg":
                self._var_index[arg["arg"]] = 0
        if "test" in self.ast_dict:
            self._parse_function_call(
                self.ast_dict["test"],
                func_name=self.ast_dict["test"]["func"]["id"],
                unique_func_name="test",
                f_type="While",
            )
        for node in self.ast_dict.get("body", []):
            self._visit_node(node)
        return self.graph, self.function_defs

    def _visit_node(self, node):
        if node["_type"] == "Assign":
            self._handle_assign(node)
        elif node["_type"] == "Expr":
            self._handle_expr(node)
        elif node["_type"] == "While":
            self._handle_while(node)
        elif node["_type"] == "For":
            self._handle_for(node)

    def _handle_while(self, node):
        if node["test"]["_type"] != "Call":
            raise NotImplementedError("Only function calls allowed in while test")
        func = self.scope[node["test"]["func"]["id"]]
        while_dict = {
            "test": _to_node_dict_entry(
                func, parse_input_args(func), {"output": parse_output_args(func)}
            )
        }
        output_vars, input_vars = _extract_variables_from_ast_body(node)
        graph, f_dict = FunctionDictFlowAnalyzer(
            node, self.scope, input_vars=input_vars
        ).analyze()
        output_counts = _get_output_counts(graph)
        nodes = _get_nodes(f_dict, output_counts)
        data_edges = _get_data_edges(graph, f_dict, output_vars, nodes)
        unique_func_name = self._get_unique_func_name("injected_while_loop")
        while_dict.update(
            _to_workflow_dict_entry(
                inputs={key: {} for key in input_vars},
                outputs={key: {} for key in output_vars},
                nodes=nodes,
                data_edges=data_edges,
                label=unique_func_name,
            )
        )
        self.function_defs[unique_func_name] = {
            "function": InjectedLoop(while_dict),
            "type": "Assign",
        }

    def _handle_for(self, node):
        if node["iter"]["_type"] != "Call":
            raise NotImplementedError("Only function calls allowed in while test")
        func = self.scope[node["iter"]["func"]["id"]]
        for_dict = {
            "iter": _to_node_dict_entry(
                func, parse_input_args(func), {"output": parse_output_args(func)}
            )
        }
        output_vars, input_vars = _extract_variables_from_ast_body(node)
        input_vars = list(input_vars) + [node["iter"]["args"][0]["id"]]
        graph, f_dict = FunctionDictFlowAnalyzer(
            node, self.scope, input_vars=input_vars
        ).analyze()
        output_counts = _get_output_counts(graph)
        nodes = _get_nodes(f_dict, output_counts)
        data_edges = _get_data_edges(graph, f_dict, output_vars)
        unique_func_name = self._get_unique_func_name("injected_for_loop")
        for_dict.update(
            _to_workflow_dict_entry(
                inputs={key: {} for key in input_vars},
                outputs={key: {} for key in output_vars},
                nodes=nodes,
                data_edges=data_edges,
                label=unique_func_name,
            )
        )
        self.function_defs[unique_func_name] = {
            "function": InjectedLoop(for_dict),
            "type": "Assign",
        }


    def _handle_expr(self, node):
        value = node["value"]
        return self._parse_function_call(value)

    def _parse_function_call(
        self, value, func_name=None, unique_func_name=None, f_type="Assign"
    ):
        if value["_type"] != "Call":
            raise NotImplementedError("Only function calls allowed on RHS")

        func_node = value["func"]
        if func_node["_type"] != "Name":
            raise NotImplementedError("Only simple functions allowed")

        if func_name is None:
            func_name = func_node["id"]
        if unique_func_name is None:
            unique_func_name = self._get_unique_func_name(func_name)

        if func_name not in self.scope:
            raise ValueError(f"Function {func_name} not found in scope")

        self.function_defs[unique_func_name] = {
            "function": self.scope[func_name],
            "type": f_type,
        }

        # Parse inputs (positional + keyword)
        for i, arg in enumerate(value.get("args", [])):
            self._add_input_edge(arg, unique_func_name, input_index=i)
        for kw in value.get("keywords", []):
            self._add_input_edge(kw["value"], unique_func_name, input_name=kw["arg"])
        return unique_func_name

    def _handle_assign(self, node):
        unique_func_name = self._handle_expr(node)
        # Parse outputs
        self._parse_outputs(node["targets"], unique_func_name)

    def _parse_outputs(self, targets, unique_func_name):
        if len(targets) == 1 and targets[0]["_type"] == "Tuple":
            for idx, elt in enumerate(targets[0]["elts"]):
                self._add_output_edge(unique_func_name, elt, output_index=idx)
        else:
            for target in targets:
                self._add_output_edge(unique_func_name, target)

    def _add_output_edge(self, source, target, **kwargs):
        var_name = target["id"]
        self._var_index[var_name] = self._var_index.get(var_name, -1) + 1
        versioned = f"{var_name}_{self._var_index[var_name]}"
        self.graph.add_edge(source, versioned, type="output", **kwargs)

    def _add_input_edge(self, source, target, **kwargs):
        if source["_type"] != "Name":
            raise NotImplementedError(f"Only variable inputs supported, got: {source}")
        var_name = source["id"]
        if var_name not in self._var_index:
            raise ValueError(f"Variable {var_name} not found in scope")
        idx = self._var_index[var_name]
        versioned = f"{var_name}_{idx}"
        self.graph.add_edge(versioned, target, type="input", **kwargs)

    def _get_unique_func_name(self, base_name):
        i = self._call_counter.get(base_name, 0)
        self._call_counter[base_name] = i + 1
        return f"{base_name}_{i}"


def get_ast_dict(func: callable) -> dict:
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
    return dict(zip(var_output, data_output))


def _get_node_outputs(func, counts):
    output_hints = parse_output_args(func, separate_tuple=counts > 1)
    output_vars = get_return_expressions(func)
    if output_vars is None or len(output_vars) == 0:
        return {}
    if counts == 1:
        if isinstance(output_vars, str):
            return {output_vars: output_hints}
        else:
            return {"output": output_hints}
    assert isinstance(output_vars, tuple) and len(output_vars) == counts
    if output_hints == {}:
        output_hints = counts * [{}]
    assert len(output_vars) == len(output_hints)
    return {key: value for key, value in zip(output_vars, output_hints)}


def _get_output_counts(graph: nx.DiGraph) -> dict:
    """
    Get the number of outputs for each node in the graph.

    Args:
        graph (nx.DiGraph): The directed graph representing the function.

    Returns:
        dict: A dictionary mapping node names to the number of outputs.
    """
    f_dict = {}
    for edge in graph.edges.data():
        if edge[2]["type"] != "output":
            continue
        f_dict[edge[0]] = f_dict.get(edge[0], 0) + 1
    return f_dict


def _get_nodes(data, output_counts):
    result = {}
    for node, function in data.items():
        if function["type"] != "Assign":
            continue
        func = function["function"]
        if hasattr(func, "_semantikon_workflow"):
            data_dict = func._semantikon_workflow.copy()
            result[node] = data_dict
            result[node]["label"] = node
        else:
            result[node] = _to_node_dict_entry(
                func,
                parse_input_args(func),
                _get_node_outputs(func, output_counts.get(node, 0)),
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


def _get_data_edges(graph, functions, output_labels, nodes):
    input_dict = {}
    for name, func in functions.items():
        f = func["function"]
        if hasattr(f, "_semantikon_workflow"):
            input_dict[name] = list(f._semantikon_workflow["inputs"].keys())
        else:
            input_dict[name] = list(parse_input_args(f).keys())
    data_edges = []
    output_dict = {}
    output_candidate = {}
    for edge in _get_sorted_edges(graph):
        if edge[2]["type"] == "output":
            if hasattr(functions[edge[0]]["function"], "_semantikon_workflow"):
                keys = list(
                    functions[edge[0]]["function"]
                    ._semantikon_workflow["outputs"]
                    .keys()
                )
                output_key = keys[0]
                if "output_index" in edge[2]:
                    output_key = keys[edge[2]["output_index"]]
            elif "output_index" in edge[2]:
                output_key = list(nodes[edge[0]]["outputs"].keys())[
                    edge[2]["output_index"]
                ]
            else:
                output_key = list(nodes[edge[0]]["outputs"].keys())[0]
            tag = f"{edge[0]}.outputs.{output_key}"
            if _remove_index(edge[1]) in output_labels:
                output_candidate[_remove_index(edge[1])] = (
                    tag,
                    f"outputs.{_remove_index(edge[1])}",
                )
            output_dict[edge[1]] = tag
        else:
            if edge[0] not in output_dict:
                source = f"inputs.{_remove_index(edge[0])}"
            else:
                source = output_dict[edge[0]]
            if "input_name" in edge[2]:
                target = f"{edge[1]}.inputs.{edge[2]['input_name']}"
            elif "input_index" in edge[2]:
                target = (
                    f"{edge[1]}.inputs.{input_dict[edge[1]][edge[2]['input_index']]}"
                )
            data_edges.append((source, target))
    for edges in output_candidate.values():
        data_edges.append(edges)
    return data_edges


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
        func (callable): The function to be analyzed.
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


def separate_types(data, class_dict=None):
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


def separate_functions(data, function_dict=None):
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


def _to_node_dict_entry(function: callable, inputs: dict, outputs: dict) -> dict:
    assert isinstance(inputs, dict)
    assert all(isinstance(v, dict) for v in inputs.values())
    assert isinstance(outputs, dict)
    assert all(isinstance(v, dict) for v in outputs.values())
    assert callable(function)
    return {"function": function, "inputs": inputs, "outputs": outputs}


def _to_workflow_dict_entry(inputs, outputs, nodes, data_edges, label, **kwargs):
    assert isinstance(inputs, dict)
    assert all(isinstance(v, dict) for v in inputs.values())
    assert isinstance(outputs, dict)
    assert all(isinstance(v, dict) for v in outputs.values())
    assert isinstance(nodes, dict)
    assert all(isinstance(v, dict) for v in nodes.values())
    assert all("inputs" in v for v in nodes.values())
    assert all("outputs" in v for v in nodes.values())
    assert all(
        "function" in v or ("nodes" in v and "data_edges" in v) for v in nodes.values()
    )
    assert isinstance(data_edges, list)
    assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in data_edges)
    assert isinstance(label, str)
    return {
        "inputs": inputs,
        "outputs": outputs,
        "nodes": nodes,
        "data_edges": data_edges,
        "label": label,
    } | kwargs


def get_workflow_dict(func):
    graph, f_dict = analyze_function(func)
    output_counts = _get_output_counts(graph)
    output_labels = list(_get_workflow_outputs(func).keys())
    nodes = _get_nodes(f_dict, output_counts)
    return _to_workflow_dict_entry(
        inputs=parse_input_args(func),
        outputs=_get_workflow_outputs(func),
        nodes=_get_nodes(f_dict, output_counts),
        data_edges=_get_data_edges(graph, f_dict, output_labels, nodes),
        label=func.__name__,
    )


def _get_missing_edges(edges):
    extra_edges = []
    for edge in edges:
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
    def __init__(self, workflow_dict):
        self._workflow = workflow_dict

    @cached_property
    def _all_edges(self):
        extra_edges = _get_missing_edges(self._workflow["data_edges"])
        return self._workflow["data_edges"] + extra_edges

    @cached_property
    def _graph(self):
        graph = nx.DiGraph()
        for edge in self._all_edges:
            graph.add_edge(*edge)
        return graph

    @cached_property
    def _execution_list(self):
        return find_parallel_execution_levels(self._graph)

    def _sanitize_input(self, *args, **kwargs):
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

    def _get_value_from_data(self, node):
        if "value" not in node:
            node["value"] = node["default"]
        return node["value"]

    def _get_value_from_global(self, path):
        io, var = path.split(".")
        return self._get_value_from_data(self._workflow[io][var])

    def _get_value_from_node(self, path):
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

    def _execute_node(self, function):
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

    def _get_value(self, tag):
        if len(tag.split(".")) == 2 and tag.split(".")[0] in ("inputs", "outputs"):
            return self._get_value_from_global(tag)
        elif len(tag.split(".")) == 3 and tag.split(".")[1] in ("inputs", "outputs"):
            return self._get_value_from_node(tag)
        elif "." not in tag:
            return self._execute_node(tag)
        else:
            raise ValueError(f"{tag} not recognized")

    def run(self, *args, **kwargs):
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


def find_parallel_execution_levels(G):
    in_degree = dict(G.in_degree())  # Track incoming edges
    queue = deque([node for node in G.nodes if in_degree[node] == 0])
    levels = []

    while queue:
        current_level = list(queue)
        levels.append(current_level)

        next_queue = deque()
        for node in current_level:
            for neighbor in G.successors(node):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_queue.append(neighbor)

        queue = next_queue

    return levels


def workflow(func):
    workflow_dict = get_workflow_dict(func)
    w = _Workflow(workflow_dict)
    func._semantikon_workflow = workflow_dict
    func.run = w.run
    return func
