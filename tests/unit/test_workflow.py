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


class TestSnippets(unittest.TestCase):
    def test_number_to_letter(self):
        self.assertEqual(number_to_letter(0), "A")
        self.assertEqual(number_to_letter(1), "B")
        self.assertRaises(ValueError, number_to_letter, -1)

    def test_analyzer(self):
        analyzer = analyze_function(example_macro)
        all_data = [
            ("operation_0", "c", {"type": "output", "output_index": 0}),
            ("operation_0", "d", {"type": "output", "output_index": 1}),
            ("c", "add_0", {"type": "input", "input_index": 0}),
            ("d", "add_0", {"type": "input", "input_name": "y"}),
            ("a", "operation_0", {"type": "input", "input_index": 0}),
            ("b", "operation_0", {"type": "input", "input_index": 1}),
            ("add_0", "e", {"type": "output"}),
            ("e", "multiply_0", {"type": "input", "input_index": 0}),
            ("multiply_0", "f", {"type": "output"}),
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
            "input": {"a": {"default": 10}, "b": {"default": 20}},
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
                ["operation_0.outputs.output_0", "add_0.inputs.x"],
                ["operation_0.outputs.output_1", "add_0.inputs.y"],
                ["inputs.a", "operation_0.inputs.x"],
                ["inputs.b", "operation_0.inputs.y"],
                ["add_0.outputs.output", "multiply_0.inputs.x"],
                ["multiply_0.outputs.output", "outputs.f"],
            ],
            "label": "example_macro",
        }
        self.assertEqual(get_workflow_dict(example_macro), ref_data)
        self.assertEqual(example_macro._semantikon_workflow, ref_data)

    def test_get_workflow_dict_macro(self):
        ref_data = {
            "input": {"a": {"default": 10}, "b": {"default": 20}},
            "outputs": {"z": {}},
            "nodes": {
                "example_macro_0": {
                    "input": {"a": {"default": 10}, "b": {"default": 20}},
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
                        ["operation_0.outputs.output_0", "add_0.inputs.x"],
                        ["operation_0.outputs.output_1", "add_0.inputs.y"],
                        ["inputs.a", "operation_0.inputs.x"],
                        ["inputs.b", "operation_0.inputs.y"],
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
                ["example_macro_0.outputs.output", "add_0.inputs.x"],
                ["inputs.a", "example_macro_0.inputs.a"],
                ["inputs.b", "example_macro_0.inputs.b"],
                ["add_0.outputs.output", "outputs.z"],
            ],
            "label": "example_workflow",
        }
        self.assertEqual(get_workflow_dict(example_workflow), ref_data)

    def test_parallel_execution(self):
        analyzer = analyze_function(parallel_execution)
        self.assertEqual(
            find_parallel_execution_levels(analyzer.graph),
            [
                ["a", "b"],
                ["add_0", "multiply_0"],
                ["c", "d"],
                ["operation_0"],
                ["e", "f"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
