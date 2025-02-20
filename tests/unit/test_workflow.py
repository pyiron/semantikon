import unittest
from semantikon.workflow import (
    number_to_letter,
    analyze_function,
    get_return_variables,
    _get_output_counts,
    get_workflow_dict,
)


def operation(x: float, y: float) -> tuple[float, float]:
    return x + y, x - y


def add(x: float = 2.0, y: float = 1) -> float:
    return x + y


def multiply(x: float, y: float) -> float:
    return x * y


def example_function(a=10, b=20):
    c, d = operation(a, b)
    e = add(c, y=d)
    f = multiply(e, 5)
    return f


class TestSnippets(unittest.TestCase):
    def test_number_to_letter(self):
        self.assertEqual(number_to_letter(0), "A")
        self.assertEqual(number_to_letter(1), "B")
        self.assertRaises(ValueError, number_to_letter, -1)

    def test_analyzer(self):
        analyzer = analyze_function(example_function)
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
        self.assertEqual(get_return_variables(example_function), ["f"])
        self.assertRaises(ValueError, get_return_variables, add)
        self.assertRaises(ValueError, get_return_variables, operation)

    def test_get_output_counts(self):
        analyzer = analyze_function(example_function)
        output_counts = _get_output_counts(analyzer.graph.edges.data())
        self.assertEqual(output_counts, {"operation": 2, "add": 1, "multiply": 1})

    def test_get_workflow_dict(self):
        ref_data = {
            "input": {"a": {"default": 10}, "b": {"default": 20}},
            "outputs": {"f": {}},
            "nodes": {
                "operation": {
                    "inputs": {"x": {"dtype": float}, "y": {"dtype": float}},
                    "outputs": {
                        "output_0": {"dtype": float},
                        "output_1": {"dtype": float},
                    },
                    "function": operation,
                },
                "add": {
                    "inputs": {
                        "x": {"dtype": float, "default": 2.0},
                        "y": {"dtype": float, "default": 1},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "function": add,
                },
                "multiply": {
                    "inputs": {"x": {"dtype": float}, "y": {"dtype": float}},
                    "outputs": {"output": {"dtype": float}},
                    "function": multiply,
                },
            },
        }
        self.assertEqual(get_workflow_dict(example_function), ref_data)


if __name__ == "__main__":
    unittest.main()
