import unittest
from semantikon.typing import u
from semantikon.workflow import (
    analyze_function,
    get_return_variables,
    _get_output_counts,
    get_workflow_dict,
    workflow,
    find_parallel_execution_levels,
    get_node_dict,
)
import numpy as np


def operation(x: float, y: float) -> tuple[float, float]:
    return x + y, x - y


@u(uri="add")
def add(x: float = 2.0, y: float = 1) -> float:
    return x + y


def multiply(x: float, y: float = 5) -> float:
    return x * y


@workflow
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


class ApeClass:
    pass


@u(uri="my_function")
def multiple_types_for_ape(a: ApeClass, b: ApeClass) -> ApeClass:
    return a + b


class TestWorkflow(unittest.TestCase):
    def test_analyzer(self):
        analyzer = analyze_function(example_macro)
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
        ]
        self.assertEqual(
            [data for data in analyzer.graph.edges.data()],
            all_data,
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
            },
        )

    def test_get_return_variables(self):
        self.assertEqual(get_return_variables(example_macro), ["f"])
        with self.assertWarns(SyntaxWarning):
            self.assertEqual(get_return_variables(add), ["output"])
        self.assertRaises(ValueError, get_return_variables, operation)

    def test_get_output_counts(self):
        analyzer = analyze_function(example_macro)
        output_counts = _get_output_counts(analyzer.graph.edges.data())
        self.assertEqual(output_counts, {"operation": 2, "add": 1, "multiply": 1})

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
                    "function": operation,
                },
                "add_0": {
                    "inputs": {
                        "x": {"dtype": float, "default": 2.0},
                        "y": {"dtype": float, "default": 1},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "function": add,
                    "uri": "add",
                },
                "multiply_0": {
                    "inputs": {
                        "x": {"dtype": float},
                        "y": {"dtype": float, "default": 5},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "function": multiply,
                },
            },
            "data_edges": [
                ["inputs.a", "operation_0.inputs.x"],
                ["inputs.b", "operation_0.inputs.y"],
                ["operation_0.outputs.output_0", "add_0.inputs.x"],
                ["operation_0.outputs.output_1", "add_0.inputs.y"],
                ["add_0.outputs.output", "multiply_0.inputs.x"],
                ["multiply_0.outputs.output", "outputs.f"],
            ],
            "label": "example_macro",
        }
        self.assertEqual(get_workflow_dict(example_macro), ref_data)
        self.assertEqual(example_macro._semantikon_workflow, ref_data)

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
                            "function": operation,
                            "inputs": {"x": {"dtype": float}, "y": {"dtype": float}},
                            "outputs": {
                                "output_0": {"dtype": float},
                                "output_1": {"dtype": float},
                            },
                        },
                        "add_0": {
                            "function": add,
                            "inputs": {
                                "x": {"dtype": float, "default": 2.0},
                                "y": {"dtype": float, "default": 1},
                            },
                            "outputs": {"output": {"dtype": float}},
                            "uri": "add",
                        },
                        "multiply_0": {
                            "function": multiply,
                            "inputs": {
                                "x": {"dtype": float},
                                "y": {"dtype": float, "default": 5},
                            },
                            "outputs": {"output": {"dtype": float}},
                        },
                    },
                    "data_edges": [
                        ["inputs.a", "operation_0.inputs.x"],
                        ["inputs.b", "operation_0.inputs.y"],
                        ["operation_0.outputs.output_0", "add_0.inputs.x"],
                        ["operation_0.outputs.output_1", "add_0.inputs.y"],
                        ["add_0.outputs.output", "multiply_0.inputs.x"],
                        ["multiply_0.outputs.output", "outputs.f"],
                    ],
                    "label": "example_macro_0",
                },
                "add_0": {
                    "function": add,
                    "inputs": {
                        "x": {"dtype": float, "default": 2.0},
                        "y": {"dtype": float, "default": 1},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "uri": "add",
                },
            },
            "data_edges": [
                ["inputs.a", "example_macro_0.inputs.a"],
                ["inputs.b", "example_macro_0.inputs.b"],
                ["inputs.b", "add_0.inputs.y"],
                ["example_macro_0.outputs.f", "add_0.inputs.x"],
                ["add_0.outputs.output", "outputs.z"],
            ],
            "label": "example_workflow",
        }
        self.assertEqual(result, ref_data, msg=result)

    def test_parallel_execution(self):
        analyzer = analyze_function(parallel_execution)
        self.assertEqual(
            find_parallel_execution_levels(analyzer.graph),
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
        self.assertEqual(parallel_execution(), data["outputs"]["e"]["value"])

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


if __name__ == "__main__":
    unittest.main()
