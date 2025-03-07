import ast
import networkx as nx
from networkx.algorithms.dag import topological_sort
import inspect
from collections import deque
from functools import cached_property
import warnings
from hashlib import sha256

from semantikon.converter import parse_input_args, parse_output_args


def _hash_function(func):
    return sha256(inspect.getsource(func).encode()).hexdigest()


def _check_node(node):
    if isinstance(node.value, (ast.BinOp, ast.Call, ast.Attribute, ast.Subscript)):
        warnings.warn(
            "Return statement contains an operation or function call, replaced"
            " by `output`",
            SyntaxWarning,
        )
        return ["output"]
    elif isinstance(node.value, ast.Tuple):  # If returning multiple values
        for elt in node.value.elts:
            if not isinstance(elt, ast.Name):
                raise ValueError(f"Invalid return value: {ast.dump(elt)}")
        return [elt.id for elt in node.value.elts if isinstance(elt, ast.Name)]

    elif isinstance(node.value, ast.Name):  # If returning a single variable
        return [node.value.id]

    else:
        raise ValueError(f"Invalid return value: {ast.dump(node.value)}")


def get_return_variables(func):
    source = ast.parse(inspect.getsource(func))
    for node in ast.walk(source):
        if not isinstance(node, ast.Return):
            continue
        return _check_node(node)
    return []


class FunctionFlowAnalyzer(ast.NodeVisitor):
    def __init__(self, scope):
        self.graph = nx.DiGraph()
        self.function_defs = {}
        self.scope = scope
        self._var_index = {}

    @staticmethod
    def _is_variable(arg):
        return isinstance(arg, ast.Name)

    def _parse_outputs(self, node, called_func):
        is_multi_outputs = len(node.targets) == 1 and isinstance(
            node.targets[0], ast.Tuple
        )
        if is_multi_outputs:
            for index, target in enumerate(node.targets[0].elts):
                self._add_output_edge(called_func, target, output_index=index)
        else:
            for target in node.targets:
                self._add_output_edge(called_func, target)

    def _add_output_edge(self, source, target, **kwargs):
        if target.id not in self._var_index:
            self._var_index[target.id] = 0
        else:
            self._var_index[target.id] += 1
        count = self._var_index[target.id]
        self.graph.add_edge(source, f"{target.id}_{count}", type="output", **kwargs)

    def _parse_inputs(self, node, called_func):
        for index, arg in enumerate(node.value.args):
            self._add_input_edge(arg, called_func, input_index=index)
        for kw in node.value.keywords:
            self._add_input_edge(kw.value, called_func, input_name=kw.arg)

    def _add_input_edge(self, source, target, **kwargs):
        if not self._is_variable(source):
            raise NotImplementedError(f"Invalid input: {ast.dump(source)}")
        if source.id not in self._var_index:
            tag = f"{source.id}_0"
        else:
            tag = f"{source.id}_{self._var_index[source.id]}"
        self.graph.add_edge(tag, target, type="input", **kwargs)

    def _get_func_name(self, node):
        for ii in range(100):
            if f"{node.value.func.id}_{ii}" not in self.graph:
                return f"{node.value.func.id}_{ii}"
        raise AssertionError("Too many times function used")

    def visit_Assign(self, node):
        """Handles variable assignments including tuple unpacking."""
        if not isinstance(node.value, ast.Call) or not isinstance(
            node.value.func, ast.Name
        ):
            raise NotImplementedError("Only function calls are allowed in assignments")
        called_func = self._get_func_name(node)
        if node.value.func.id not in self.scope:
            raise ValueError(f"Function {node.value.func.id} not defined")
        self.function_defs[called_func] = self.scope[node.value.func.id]
        self._parse_outputs(node, called_func)
        self._parse_inputs(node, called_func)
        self.generic_visit(node)


def analyze_function(func):
    """Extracts the variable flow graph from a function"""
    source_code = inspect.getsource(func)
    scope = inspect.getmodule(func).__dict__
    tree = ast.parse(source_code)
    analyzer = FunctionFlowAnalyzer(scope)
    analyzer.visit(tree)
    return analyzer


def _get_workflow_outputs(func):
    var_output = get_return_variables(func)
    data_output = parse_output_args(func)
    if isinstance(data_output, dict):
        data_output = [data_output]
    return dict(zip(var_output, data_output))


def _get_node_outputs(func, counts):
    outputs = parse_output_args(func)
    if outputs == {} and counts > 1:
        outputs = counts * [{}]
    if isinstance(outputs, tuple):
        return {f"output_{ii}": v for ii, v in enumerate(outputs)}
    else:
        return {"output": outputs}


def _get_output_counts(data):
    f_dict = {}
    for edge in data:
        if edge[2]["type"] != "output":
            continue
        f_name = "_".join(edge[0].split("_")[:-1])
        if f_name not in f_dict or f_dict[f_name] < edge[2].get("output_index", 0) + 1:
            f_dict[f_name] = edge[2].get("output_index", 0) + 1
    return f_dict


def _get_nodes(data, output_counts):
    result = {}
    for node, func in data.items():
        if hasattr(func, "_semantikon_workflow"):
            result[node] = func._semantikon_workflow.copy()
            result[node]["label"] = node
        else:
            result[node] = {
                "function": func,
                "inputs": parse_input_args(func),
                "outputs": _get_node_outputs(func, output_counts.get(node, 1)),
            }
    return result


def _remove_index(s):
    return "_".join(s.split("_")[:-1])


def _get_sorted_edges(graph):
    topo_order = list(topological_sort(graph))
    node_order = {node: i for i, node in enumerate(topo_order)}
    return sorted(graph.edges.data(), key=lambda edge: node_order[edge[0]])


def _get_data_edges(analyzer, func):
    input_dict = {
        name: list(parse_input_args(func).keys())
        for name, func in analyzer.function_defs.items()
    }
    output_labels = list(_get_workflow_outputs(func).keys())
    data_edges = []
    output_dict = {}
    ordered_edges = _get_sorted_edges(analyzer.graph)
    for edge in ordered_edges:
        if edge[2]["type"] == "output":
            if hasattr(analyzer.function_defs[edge[0]], "_semantikon_workflow"):
                keys = list(
                    analyzer.function_defs[edge[0]]
                    ._semantikon_workflow["outputs"]
                    .keys()
                )
                output_index = 0
                if "output_index" in edge[2]:
                    output_index = edge[2]["output_index"]
                tag = f"{edge[0]}.outputs.{keys[output_index]}"
            elif "output_index" in edge[2]:
                tag = f"{edge[0]}.outputs.output_{edge[2]['output_index']}"
            else:
                tag = f"{edge[0]}.outputs.output"
            if _remove_index(edge[1]) in output_labels:
                data_edges.append([tag, f"outputs.{_remove_index(edge[1])}"])
            else:
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
            data_edges.append([source, target])
    return data_edges


def _dtype_to_ape_format(dtype):
    standard_dict = {str: "String", float: "Float", bool: "Bool", int: "Int"}
    if dtype in standard_dict:
        return standard_dict[dtype]
    else:
        return dtype.__module__ + "." + dtype.__name__


def _to_ape(data, func):
    try:
        data["taxonomyOperations"] = [data.pop("uri")]
    except KeyError:
        raise KeyError(f"uri not found in metadata of {func.__name__}")
    data["id"] = data["label"] + "_" + _hash_function(func)
    for io_ in ["inputs", "outputs"]:
        io_data = []
        for key, value in data[io_].items():
            try:
                io_data.append(
                    {
                        "Type": value["uri"],
                        "Format": _dtype_to_ape_format(value["dtype"]),
                    }
                )
            except KeyError:
                raise KeyError(
                    f"uri not found in {io_} metadata of {func.__name__} {key}"
                )
        data[io_] = io_data
    return data


def get_node_dict(func, data_format="semantikon"):
    """
    Get a dictionary representation of the function node.

    Args:
        func (callable): The function to be analyzed.
        data_format (str): The format of the output. Options are "semantiokn" and
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


def get_workflow_dict(func):
    analyzer = analyze_function(func)
    output_counts = _get_output_counts(analyzer.graph.edges.data())
    data = {
        "inputs": parse_input_args(func),
        "outputs": _get_workflow_outputs(func),
        "nodes": _get_nodes(analyzer.function_defs, output_counts),
        "data_edges": _get_data_edges(analyzer, func),
        "label": func.__name__,
    }
    return data


def _get_missing_edges(edges):
    extra_edges = []
    for edge in edges:
        for tag in edge:
            if len(tag.split(".")) < 3:
                continue
            if tag.split(".")[1] == "inputs":
                new_edge = [tag, tag.split(".")[0]]
            elif tag.split(".")[1] == "outputs":
                new_edge = [tag.split(".")[0], tag]
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
