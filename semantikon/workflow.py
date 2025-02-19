import ast
import networkx as nx
import inspect

class FunctionFlowAnalyzer(ast.NodeVisitor):
    def __init__(self, scope):
        self.graph = nx.DiGraph()
        self.function_defs = {}
        self.scope = scope

    def visit_Assign(self, node):
        """Handles variable assignments including tuple unpacking."""
        if isinstance(node.value, ast.Call):  # Function call in assignment
            if isinstance(node.value.func, ast.Name):  # Direct function call
                for ii in range(100):
                    if f"{node.value.func.id}_{ii}" not in self.graph:
                        called_func = f"{node.value.func.id}_{ii}"
                        break
                else:
                    raise AssertionError("Too many times function used")
                if node.value.func.id not in self.scope:
                    raise ValueError(f"Function {node.value.func.id} not defined")
                self.function_defs[called_func] = self.scope[node.value.func.id]

                # Track function outputs: handle single or multiple assignments
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
                    # Tuple unpacking case (e.g., c, d = func(...))
                    for index, target in enumerate(node.targets[0].elts):
                        if isinstance(target, ast.Name):
                            self.graph.add_edge(called_func, target.id, type="output", output_index=index)
                else:
                    # Single assignment case (e.g., x = func(...))
                    for target in node.targets:
                        if isinstance(target, ast.Name):  # Ensure it's a variable
                            self.graph.add_edge(called_func, target.id, type="output")

                # Track positional arguments
                for index, arg in enumerate(node.value.args):
                    if isinstance(arg, ast.Name):  # If argument is a variable
                        self.graph.add_edge(arg.id, called_func, type="input", input_index=index)

                # Track keyword arguments
                for kw in node.value.keywords:
                    if isinstance(kw.value, ast.Name):  # If the value is a variable
                        self.graph.add_edge(kw.value.id, called_func, type="input", input_name=kw.arg)

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
        return chr(n + ord('A'))
    else:
        raise ValueError("Number out of range")
