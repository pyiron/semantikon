import ast
import networkx as nx
from networkx.algorithms.dag import topological_sort
import inspect
from collections import deque

from semantikon.converter import parse_input_args, parse_output_args



def _check_node(node):
    if isinstance(node.value, (ast.BinOp, ast.Call, ast.Attribute, ast.Subscript)):
        raise ValueError("Return statement contains an operation or function call.")


def get_return_variables(func):
    source = ast.parse(inspect.getsource(func))
    for node in ast.walk(source):
        if not isinstance(node, ast.Return):
            continue
        _check_node(node)
        if isinstance(node.value, ast.Tuple):  # If returning multiple values
            for elt in node.value.elts:
                if not isinstance(elt, ast.Name):
                    raise ValueError(f"Invalid return value: {ast.dump(elt)}")
            return [elt.id for elt in node.value.elts if isinstance(elt, ast.Name)]

        elif isinstance(node.value, ast.Name):  # If returning a single variable
            return [node.value.id]

        else:
            raise ValueError(f"Invalid return value: {ast.dump(node.value)}")

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

    def _add_output_edge(self, source, target, **kwargs):
        if self._is_variable(target):
            if target.id not in self._var_index:
                self._var_index[target.id] = 0
            else:
                self._var_index[target.id] += 1
            count = self._var_index[target.id]
            self.graph.add_edge(
                source, f"{target.id}_{count}", type="output", **kwargs
            )

    def _add_input_edge(self, source, target, **kwargs):
        if self._is_variable(source):
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
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            called_func = self._get_func_name(node)
            if node.value.func.id not in self.scope:
                raise ValueError(f"Function {node.value.func.id} not defined")
            self.function_defs[called_func] = self.scope[node.value.func.id]

            is_multi_assignment = len(node.targets) == 1 and isinstance(
                node.targets[0], ast.Tuple
            )

            if is_multi_assignment:
                for index, target in enumerate(node.targets[0].elts):
                    self._add_output_edge(called_func, target, output_index=index)
            else:
                for target in node.targets:
                    self._add_output_edge(called_func, target)

            for index, arg in enumerate(node.value.args):
                self._add_input_edge(arg, called_func, input_index=index)
            for kw in node.value.keywords:
                self._add_input_edge(kw.value, called_func, input_name=kw.arg)

        self.generic_visit(node)


def analyze_function(func):
    """Extracts the variable flow graph from a function"""
    source_code = inspect.getsource(func)
    scope = inspect.getmodule(func).__dict__
    tree = ast.parse(source_code)
    analyzer = FunctionFlowAnalyzer(scope)
    analyzer.visit(tree)
    return analyzer


def number_to_letter(n):
    if 0 <= n <= 25:
        return chr(n + ord("A"))
    else:
        raise ValueError("Number out of range")


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
            result[node] = func._semantikon_workflow
        else:
            result[node] = {
                "function": func,
                "inputs": parse_input_args(func),
                "outputs": _get_node_outputs(func, output_counts.get(node, 1)),
            }
    return result


def _remove_index(s):
    return "_".join(s.split("_")[:-1])


def get_sorted_edges(graph):
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
    ordered_edges = get_sorted_edges(analyzer.graph)
    for edge in ordered_edges:
        if edge[2]["type"] == "output":
            if "output_index" in edge[2]:
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
                target = f"{edge[1]}.inputs.{input_dict[edge[1]][edge[2]['input_index']]}"
            data_edges.append([source, target])
    return data_edges


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


def _get_execution_list(edges):
    extra_edges = _get_missing_edges(edges)
    graph = nx.DiGraph()
    for edge in edges + extra_edges:
        graph.add_edge(*edge)
    return find_parallel_execution_levels(graph)


class _Workflow:
    def __init__(self, func):
        self._workflow = get_workflow_dict(func)

    def get_workflow_dict(self):
        return self._workflow

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

    def run(self, *args, **kwargs):
        self._set_inputs(*args, **kwargs)
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
    w = _Workflow(func)
    func._semantikon_workflow = w.get_workflow_dict()
    func.run = w.run
    return func
