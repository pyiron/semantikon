import unittest
from semantikon.workflow import (
    number_to_letter,
    analyze_function,
    get_return_variables,
    _get_output_counts,
    get_workflow_dict,
    workflow,
    find_parallel_execution_levels,
)


def operation(x: float, y: float) -> tuple[float, float]:
    return x + y, x - y


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
    z = add(y, 10)
    return z


@workflow
def parallel_execution(a=10, b=20):
    c = add(a)
    d = multiply(b)
    e, f = operation(c, d)
    return e, f


class TestWorkflow(unittest.TestCase):
    def test_number_to_letter(self):
        self.assertEqual(number_to_letter(0), "A")
        self.assertEqual(number_to_letter(1), "B")
        self.assertRaises(ValueError, number_to_letter, -1)

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

    def test_get_return_variables(self):
        self.assertEqual(get_return_variables(example_macro), ["f"])
        self.assertRaises(ValueError, get_return_variables, add)
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
                    "label": "example_macro",
                },
                "add_0": {
                    "function": add,
                    "inputs": {
                        "x": {"dtype": float, "default": 2.0},
                        "y": {"dtype": float, "default": 1},
                    },
                    "outputs": {"output": {"dtype": float}},
                },
            },
            "data_edges": [
                ["inputs.a", "example_macro_0.inputs.a"],
                ["inputs.b", "example_macro_0.inputs.b"],
                ["example_macro_0.outputs.f", "add_0.inputs.x"],
                ["add_0.outputs.output", "outputs.z"],
            ],
            "label": "example_workflow",
        }
        result = get_workflow_dict(example_workflow)
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

    def test_run(self):
        data = example_macro.run()
        ref_data = {
            "inputs": {
                "a": {"default": 10, "value": 10},
                "b": {"default": 20, "value": 20},
            },
            "outputs": {"f": {"value": 100}},
            "nodes": {
                "operation_0": {
                    "function": operation,
                    "inputs": {
                        "x": {"dtype": float, "value": 10},
                        "y": {"dtype": float, "value": 20},
                    },
                    "outputs": {
                        "output_0": {"dtype": float, "value": 30},
                        "output_1": {"dtype": float, "value": -10},
                    },
                },
                "add_0": {
                    "function": add,
                    "inputs": {
                        "x": {"dtype": float, "default": 2.0, "value": 30},
                        "y": {"dtype": float, "default": 1, "value": -10},
                    },
                    "outputs": {"output": {"dtype": float, "value": 20}},
                },
                "multiply_0": {
                    "function": multiply,
                    "inputs": {
                        "x": {"dtype": float, "value": 20},
                        "y": {"dtype": float, "default": 5, "value": 5},
                    },
                    "outputs": {"output": {"dtype": float, "value": 100}},
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
        self.assertEqual(data, ref_data)
        data = parallel_execution.run()
        ref_data = {
            "inputs": {
                "a": {"default": 10, "value": 10},
                "b": {"default": 20, "value": 20},
            },
            "outputs": {"e": {"value": (111, -89)}},
            "nodes": {
                "add_0": {
                    "function": add,
                    "inputs": {
                        "x": {"dtype": float, "default": 2.0, "value": 10},
                        "y": {"dtype": float, "default": 1, "value": 1},
                    },
                    "outputs": {"output": {"dtype": float, "value": 11}},
                },
                "multiply_0": {
                    "function": multiply,
                    "inputs": {
                        "x": {"dtype": float, "value": 20},
                        "y": {"dtype": float, "default": 5, "value": 5},
                    },
                    "outputs": {"output": {"dtype": float, "value": 100}},
                },
                "operation_0": {
                    "function": operation,
                    "inputs": {
                        "x": {"dtype": float, "value": 11},
                        "y": {"dtype": float, "value": 100},
                    },
                    "outputs": {
                        "output_0": {"dtype": float, "value": (111, -89)},
                        "output_1": {"dtype": float},
                    },
                },
            },
            "data_edges": [
                ["inputs.a", "add_0.inputs.x"],
                ["inputs.b", "multiply_0.inputs.x"],
                ["add_0.outputs.output", "operation_0.inputs.x"],
                ["multiply_0.outputs.output", "operation_0.inputs.y"],
                ["operation_0.outputs.output_0", "outputs.e"],
            ],
            "label": "parallel_execution",
        }
        self.assertEqual(data, ref_data)
        data = example_workflow.run()
        print(data)


if __name__ == "__main__":
    unittest.main()
