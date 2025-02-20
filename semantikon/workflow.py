import ast
import networkx as nx
import inspect


class FunctionFlowAnalyzer(ast.NodeVisitor):
    def __init__(self, scope):
        self.graph = nx.DiGraph()
        self.function_defs = {}
        self.scope = scope

    @staticmethod
    def _is_variable(arg):
        return isinstance(arg, ast.Name)

    def visit_Assign(self, node):
        """Handles variable assignments including tuple unpacking."""
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            for ii in range(100):
                if f"{node.value.func.id}_{ii}" not in self.graph:
                    called_func = f"{node.value.func.id}_{ii}"
                    break
            else:
                raise AssertionError("Too many times function used")
            if node.value.func.id not in self.scope:
                raise ValueError(f"Function {node.value.func.id} not defined")
            self.function_defs[called_func] = self.scope[node.value.func.id]

            is_multi_assignment = len(node.targets) == 1 and isinstance(
                node.targets[0], ast.Tuple
            )

            if is_multi_assignment:
                for index, target in enumerate(node.targets[0].elts):
                    if self._is_variable(target):
                        self.graph.add_edge(
                            called_func,
                            target.id,
                            type="output",
                            output_index=index,
                        )
            else:
                for target in node.targets:
                    if self._is_variable(target):
                        self.graph.add_edge(called_func, target.id, type="output")

            # Track positional arguments
            for index, arg in enumerate(node.value.args):
                if self._is_variable(arg):
                    self.graph.add_edge(
                        arg.id, called_func, type="input", input_index=index
                    )

            # Track keyword arguments
            for kw in node.value.keywords:
                if self._is_variable(kw.value):
                    self.graph.add_edge(
                        kw.value.id,
                        called_func,
                        type="input",
                        input_name=kw.arg,
                    )

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
