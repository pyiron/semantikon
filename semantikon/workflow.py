import ast
import networkx as nx
import inspect


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

    @staticmethod
    def _is_variable(arg):
        return isinstance(arg, ast.Name)

    def _add_output_edge(self, source, target, **kwargs):
        if self._is_variable(target):
            self.graph.add_edge(source, target.id, type="output", **kwargs)

    def _add_input_edge(self, source, target, **kwargs):
        if self._is_variable(source):
            self.graph.add_edge(source.id, target, type="input", **kwargs)

    def _get_func_name(self, node):
        for ii in range(100):
            if f"{node.value.func.id}_{ii}" not in self.graph:
                return f"{node.value.func.id}_{ii}"
                break
        else:
            raise AssertionError("Too many times function used")

    def visit_Assign(self, node):
        """Handles variable assignments including tuple unpacking."""
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            called_func = self._get_func_name(node)
            if node.value.func.id not in self.scope:
                raise ValueError(f"Function {node.value.func.id} not defined")
            self.function_defs[node.value.func.id] = self.scope[node.value.func.id]

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


def _get_outputs(func):
    var_output = get_return_variables(func)
    data_output = parse_output_args(func)
    if isinstance(data_output, dict):
        data_output = [data_output]
    return dict(zip(var_output, data_output))


def _get_output_counts(data):
    f_dict = {}
    for edge in data:
        if edge[2]["type"] != "output":
            continue
        f_name = "_".join(edge[0].split("_")[:-1])
        if f_name not in f_dict or f_dict[f_name] < edge[2].get("output_index", 0) + 1:
            f_dict[f_name] = edge[2].get("output_index", 0) + 1
    return f_dict


def get_workflow_dict(func):
    data = {
        "input": parse_input_args(func),
        "outputs": _get_outputs(func),
    }
    outputs = parse_output_args(func)
    if isinstance(outputs, tuple):
        out = {f"output_{ii}": v for ii, v in enumerate(outputs)}
    else:
        out = {"output": outputs}
    {
        "function": func,
        "inputs": parse_input_args(func),
        "outputs": out,
    }
