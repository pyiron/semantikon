import ast
import unittest

import networkx as nx

from semantikon.metadata import u
from semantikon.workflow import (
    _detect_io_variables_from_control_flow,
    _extract_variables_from_ast_body,
    _function_to_ast_dict,
    _get_control_flow_graph,
    _get_node_outputs,
    _get_output_counts,
    _get_sorted_edges,
    _split_graphs_into_subgraphs,
    _get_workflow_outputs,
    analyze_function,
    ast_from_dict,
    find_parallel_execution_levels,
    get_node_dict,
    get_workflow_dict,
    separate_functions,
    separate_types,
    workflow,
)


def operation(x: float, y: float) -> tuple[float, float]:
    return x + y, x - y


@u(uri="add")
def add(x: float = 2.0, y: float = 1) -> float:
    return x + y


def multiply(x: float, y: float = 5) -> float:
    return x * y


@workflow
@u(uri="this macro has metadata")
def example_macro(a=10, b=20):
    c, d = operation(a, b)
    e = add(c, y=d)
    f = multiply(e)
    return f


@workflow
def example_workflow(a=10, b=20):
    y = example_macro(a, b)
    z = add(y, b)
    return z


@workflow
def parallel_execution(a=10, b=20):
    c = add(a)
    d = multiply(b)
    e, f = operation(c, d)
    return e, f


def example_invalid_operator(a=10, b=20):
    y = example_macro(a, b)
    z = add(y, b)
    result = z + 1
    return result


def example_invalid_multiple_operation(a=10, b=20):
    result = add(a, add(a, b))
    return result


def example_invalid_local_var_def(a=10, b=20):
    result = add(a, 2)
    return result


def my_while_condition(a=10, b=20):
    return a < b


def workflow_with_while(a=10, b=20):
    x = add(a, b)
    while my_while_condition(x, b):
        x = add(a, b)
        # Poor implementation to define variable inside the loop, but allowed
        z = multiply(a, x)
    return z


class ApeClass:
    pass


@u(uri="my_function")
def multiple_types_for_ape(a: ApeClass, b: ApeClass) -> ApeClass:
    return a + b


def seemingly_cyclic_workflow(a=10, b=20):
    a = add(a, b)
    return a


def workflow_to_use_undefined_variable(a=10, b=20):
    result = add(a, u)
    return result


def reused_args(a=10, b=20):
    a, b = operation(a, b)
    f = add(a, y=b)
    f = multiply(f)
    return f


def check_positive(x):
    if x < 0:
        raise ValueError("It must not be negative")


def workflow_with_leaf(x):
    y = add(x, x)
    check_positive(y)
    return y


class TestWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None

    def test_analyzer(self):
        graph = analyze_function(example_macro)[0]
        all_data = [
            ("operation_0", "c_0", {"type": "output", "output_index": 0}),
            ("operation_0", "d_0", {"type": "output", "output_index": 1}),
            ("c_0", "add_0", {"type": "input", "input_index": 0}),
            ("d_0", "add_0", {"type": "input", "input_name": "y"}),
            ("a_0", "operation_0", {"type": "input", "input_index": 0}),
            ("b_0", "operation_0", {"type": "input", "input_index": 1}),
            ("add_0", "e_0", {"type": "output"}),
            ("e_0", "multiply_0", {"type": "input", "input_index": 0}),
            ("multiply_0", "f_0", {"type": "output"}),
            ("f_0", "output", {"type": "input"}),
            ("input", "a_0", {"type": "output"}),
            ("input", "b_0", {"type": "output"}),
        ]
        self.maxDiff = None
        self.assertEqual(
            sorted([data for data in graph.edges.data()]),
            sorted(all_data),
        )

    def test_get_node_dict(self):
        node_dict = get_node_dict(add)
        self.assertEqual(
            node_dict,
            {
                "inputs": {
                    "x": {"dtype": float, "default": 2.0},
                    "y": {"dtype": float, "default": 1},
                },
                "outputs": {"output": {"dtype": float}},
                "label": "add",
                "uri": "add",
                "type": "Function",
            },
        )
        node_dict = get_node_dict(multiple_types_for_ape, data_format="ape")
        node_dict.pop("id")
        self.assertEqual(
            node_dict,
            {
                "inputs": [
                    {"Type": "ApeClass"},
                    {"Type": "ApeClass"},
                ],
                "outputs": [{"Type": "ApeClass"}],
                "label": "multiple_types_for_ape",
                "taxonomyOperations": ["my_function"],
                "type": "Function",
            },
        )

    def test_get_output_counts(self):
        graph = analyze_function(example_macro)[0]
        output_counts = _get_output_counts(graph)
        self.assertEqual(output_counts, {"operation_0": 2, "add_0": 1, "multiply_0": 1})

    def test_get_workflow_dict(self):
        ref_data = {
            "inputs": {"a": {"default": 10}, "b": {"default": 20}},
            "outputs": {"f": {}},
            "nodes": {
                "operation_0": {
                    "inputs": {"x": {"dtype": float}, "y": {"dtype": float}},
                    "outputs": {
                        "output_0": {"dtype": float},
                        "output_1": {"dtype": float},
                    },
                    "function": f"{operation.__module__}.operation",
                    "type": "Function",
                },
                "add_0": {
                    "inputs": {
                        "x": {"dtype": float, "default": 2.0},
                        "y": {"dtype": float, "default": 1},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "function": f"{add.__module__}.add",
                    "uri": "add",
                    "type": "Function",
                },
                "multiply_0": {
                    "inputs": {
                        "x": {"dtype": float},
                        "y": {"dtype": float, "default": 5},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "function": f"{multiply.__module__}.multiply",
                    "type": "Function",
                },
            },
            "edges": [
                ("inputs.a", "operation_0.inputs.x"),
                ("inputs.b", "operation_0.inputs.y"),
                ("operation_0.outputs.output_0", "add_0.inputs.x"),
                ("operation_0.outputs.output_1", "add_0.inputs.y"),
                ("add_0.outputs.output", "multiply_0.inputs.x"),
                ("multiply_0.outputs.output", "outputs.f"),
            ],
            "label": "example_macro",
            "type": "Workflow",
        }
        self.assertEqual(
            separate_functions(example_macro._semantikon_workflow)[0], ref_data
        )

    def test_get_workflow_dict_macro(self):
        result = get_workflow_dict(example_workflow)
        ref_data = {
            "inputs": {"a": {"default": 10}, "b": {"default": 20}},
            "outputs": {"z": {}},
            "nodes": {
                "example_macro_0": {
                    "inputs": {"a": {"default": 10}, "b": {"default": 20}},
                    "outputs": {"f": {}},
                    "nodes": {
                        "operation_0": {
                            "function": f"{operation.__module__}.operation",
                            "inputs": {"x": {"dtype": float}, "y": {"dtype": float}},
                            "outputs": {
                                "output_0": {"dtype": float},
                                "output_1": {"dtype": float},
                            },
                            "type": "Function",
                        },
                        "add_0": {
                            "function": f"{add.__module__}.add",
                            "inputs": {
                                "x": {"dtype": float, "default": 2.0},
                                "y": {"dtype": float, "default": 1},
                            },
                            "outputs": {"output": {"dtype": float}},
                            "uri": "add",
                            "type": "Function",
                        },
                        "multiply_0": {
                            "function": f"{multiply.__module__}.multiply",
                            "inputs": {
                                "x": {"dtype": float},
                                "y": {"dtype": float, "default": 5},
                            },
                            "outputs": {"output": {"dtype": float}},
                            "type": "Function",
                        },
                    },
                    "edges": [
                        ("inputs.a", "operation_0.inputs.x"),
                        ("inputs.b", "operation_0.inputs.y"),
                        ("operation_0.outputs.output_0", "add_0.inputs.x"),
                        ("operation_0.outputs.output_1", "add_0.inputs.y"),
                        ("add_0.outputs.output", "multiply_0.inputs.x"),
                        ("multiply_0.outputs.output", "outputs.f"),
                    ],
                    "label": "example_macro_0",
                    "type": "Workflow",
                    "uri": "this macro has metadata",
                },
                "add_0": {
                    "function": f"{add.__module__}.add",
                    "inputs": {
                        "x": {"dtype": float, "default": 2.0},
                        "y": {"dtype": float, "default": 1},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "uri": "add",
                    "type": "Function",
                },
            },
            "edges": [
                ("inputs.a", "example_macro_0.inputs.a"),
                ("inputs.b", "example_macro_0.inputs.b"),
                ("inputs.b", "add_0.inputs.y"),
                ("example_macro_0.outputs.f", "add_0.inputs.x"),
                ("add_0.outputs.output", "outputs.z"),
            ],
            "label": "example_workflow",
            "type": "Workflow",
        }
        self.assertEqual(separate_functions(result)[0], ref_data, msg=result)

    def test_parallel_execution(self):
        graph = analyze_function(parallel_execution)[0]
        self.assertEqual(
            find_parallel_execution_levels(graph),
            [
                ["a_0", "b_0"],
                ["add_0", "multiply_0"],
                ["c_0", "d_0"],
                ["operation_0"],
                ["e_0", "f_0"],
            ],
        )

    def test_run_single(self):
        data = example_macro.run()
        self.assertEqual(example_macro(), data["outputs"]["f"]["value"])

    def test_run_parallel_execution(self):
        data = parallel_execution.run()
        results = parallel_execution()
        self.assertEqual(results[0], data["outputs"]["e"]["value"])
        self.assertEqual(results[1], data["outputs"]["f"]["value"])

    def test_run_nested(self):
        data = example_workflow.run()
        self.assertEqual(example_workflow(), data["outputs"]["z"]["value"])

    def test_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            workflow(example_invalid_operator)
        with self.assertRaises(NotImplementedError):
            workflow(example_invalid_multiple_operation)
        with self.assertRaises(NotImplementedError):
            workflow(example_invalid_local_var_def)

    def test_separate_functions(self):
        old_data = example_workflow._semantikon_workflow
        data, function_dict = separate_functions(old_data)
        # add is deep copied due to the decorator
        del function_dict[f"{add.__module__}.add"]
        self.assertEqual(
            function_dict,
            {
                f"{operation.__module__}.operation": operation,
                f"{multiply.__module__}.multiply": multiply,
            },
        )
        self.assertEqual(
            data["nodes"]["example_macro_0"]["nodes"]["operation_0"]["function"],
            f"{operation.__module__}.operation",
        )
        self.assertEqual(
            old_data["nodes"]["example_macro_0"]["nodes"]["operation_0"]["function"],
            operation,
        )

    def test_separate_types(self):
        old_data = example_workflow._semantikon_workflow
        class_dict = separate_types(old_data)[1]
        self.assertEqual(class_dict, {"float": float})

    def test_seemingly_cyclic_workflow(self):
        data = get_workflow_dict(seemingly_cyclic_workflow)
        self.assertIn("a", data["inputs"])
        self.assertIn("a", data["outputs"])

    def test_workflow_to_use_undefined_variable(self):
        with self.assertRaises(ValueError):
            workflow(workflow_to_use_undefined_variable)

    def test_ast_from_dict(self):
        d = {
            "_type": "Compare",
            "left": {"_type": "Name", "id": "x", "ctx": {"_type": "Load"}},
            "ops": [{"_type": "Lt"}],
            "comparators": [{"_type": "Constant", "value": 0, "kind": None}],
        }
        self.assertEqual(ast.unparse(ast_from_dict(d)), "x < 0")

    def test_extract_variables_from_ast_body(self):
        body = _function_to_ast_dict(ast.parse("x = g(y)\ny = h(Z)\nz = f(x, y)"))
        variables = _extract_variables_from_ast_body(body)
        self.assertEqual(variables[0], {"x", "y", "z"})
        self.assertEqual(variables[1], {"y", "Z", "x"})

    def test_get_sorted_edges(self):
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "D"), ("A", "C"), ("C", "D")])
        sorted_edges = _get_sorted_edges(graph)
        self.assertEqual(
            sorted_edges,
            [("A", "B", {}), ("A", "C", {}), ("B", "D", {}), ("C", "D", {})],
        )

    @unittest.skip
    def test_workflow_with_while(self):
        wf = workflow(workflow_with_while)._semantikon_workflow
        self.assertIn("injected_while_loop_0", wf["nodes"])
        self.assertEqual(
            sorted(wf["nodes"]["injected_while_loop_0"]["inputs"].keys()),
            ["a", "b", "x"],
        )
        self.assertEqual(
            sorted(wf["nodes"]["injected_while_loop_0"]["outputs"].keys()),
            ["x", "z"],
        )
        self.assertEqual(
            sorted(wf["nodes"]["injected_while_loop_0"]["edges"]),
            sorted(
                [
                    ("inputs.x", "test.inputs.a"),
                    ("inputs.b", "test.inputs.b"),
                    ("inputs.b", "add_0.inputs.y"),
                    ("inputs.a", "add_0.inputs.x"),
                    ("inputs.a", "multiply_0.inputs.x"),
                    ("add_0.outputs.output", "multiply_0.inputs.y"),
                    ("add_0.outputs.output", "outputs.x"),
                    ("multiply_0.outputs.output", "outputs.z"),
                ]
            ),
        )
        self.assertIn("add_0", wf["nodes"]["injected_while_loop_0"]["nodes"])
        self.assertIn("multiply_0", wf["nodes"]["injected_while_loop_0"]["nodes"])

    def test_reused_args(self):
        data = get_workflow_dict(reused_args)
        self.assertEqual(
            sorted(data["edges"]),
            sorted(example_macro._semantikon_workflow["edges"]),
        )

    def test_get_node_outputs(self):
        self.assertEqual(
            _get_node_outputs(operation, counts=2),
            {"output_0": {"dtype": float}, "output_1": {"dtype": float}},
        )
        self.assertEqual(
            _get_node_outputs(operation, counts=1),
            {"output": {"dtype": tuple[float, float]}},
        )
        self.assertEqual(
            _get_node_outputs(parallel_execution, counts=2),
            {"e": {}, "f": {}},
        )
        self.assertEqual(
            _get_node_outputs(parallel_execution, counts=1),
            {"output": {}},
        )

    def test_workflow_with_leaf(self):
        data = get_workflow_dict(workflow_with_leaf)
        self.assertIn("check_positive_0", data["nodes"])
        self.assertIn("add_0", data["nodes"])
        self.assertIn("y", data["outputs"])
        self.assertIn(
            ("add_0.outputs.output", "check_positive_0.inputs.x"), data["edges"]
        )
        self.assertEqual(data["nodes"]["check_positive_0"]["outputs"], {})

    def test_get_workflow_output(self):

        def test_function_1(a, b):
            return a + b

        self.assertEqual(
            _get_workflow_outputs(test_function_1),
            {"output": {}},
        )

        def test_function_2(a, b):
            return a

        self.assertEqual(
            _get_workflow_outputs(test_function_2),
            {"a": {}},
        )

        def test_function_3(a, b):
            return a, b

        self.assertEqual(
            _get_workflow_outputs(test_function_3),
            {"a": {}, "b": {}},
        )

        def test_function_4(a, b):
            return a + b, b

        data = _get_workflow_outputs(test_function_4)
        self.assertEqual(data, {"output_0": {}, "b": {}})
        data["output_0"]["value"] = 0
        self.assertEqual(
            data,
            {"output_0": {"value": 0}, "b": {}},
        )

        def test_function_5(a: int, b: int) -> tuple[int, int]:
            return a, b

        self.assertEqual(
            _get_workflow_outputs(test_function_5),
            {"a": {"dtype": int}, "b": {"dtype": int}},
        )

    def test_detect_io_variables_from_control_flow(self):
        graph = analyze_function(workflow_with_while)[0]
        subgraphs = _split_graphs_into_subgraphs(graph)
        io_vars = _detect_io_variables_from_control_flow(graph, subgraphs["While_0"])
        self.assertEqual(
            {key: sorted(value) for key, value in io_vars.items()},
            {
                "inputs": ["a_0", "b_0", "x_0"],
                "outputs": ["z_0"],
            },
        )

    def test_get_control_flow_graph(self):
        control_flows = [
            "",
            "While_1/While_0",
            "While_2",
            "While_0",
            "While_1",
            "While_0/While_0/While_0",
            "While_1/While_1",
            "While_0/While_1",
            "While_0/While_0",
            "While_0/While_2",
        ]
        graph = _get_control_flow_graph(control_flows)
        self.assertEqual(
            sorted(list(graph.successors("While_0"))),
            ["While_0/While_0", "While_0/While_1", "While_0/While_2"],
        )
        self.assertEqual(
            sorted(list(graph.successors("While_1"))),
            ["While_1/While_0", "While_1/While_1"],
        )


if __name__ == "__main__":
    unittest.main()
