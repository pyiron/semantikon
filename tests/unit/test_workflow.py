import unittest
from typing import Annotated

import semantikon.workflow as swf
from semantikon import flowrep_dict
from semantikon.metadata import meta, u


def operation(x: float, y: float) -> tuple[float, float]:
    return x + y, x - y


@meta(uri="add")
def add(x: float = 2.0, y: float = 1) -> float:
    return x + y


def multiply(x: float, y: float = 5) -> float:
    return x * y


@swf.workflow
@meta(uri="this macro has metadata")
def example_macro(a=10, b=20):
    c, d = operation(a, b)
    e = add(c, y=d)
    f = multiply(e)
    return f


@swf.workflow
def example_workflow(a=10, b=20):
    y = example_macro(a, b)
    z = add(y, b)
    return z


@swf.workflow
def parallel_execution(a=10, b=20):
    c = add(a)
    d = multiply(b)
    e, f = operation(c, d)
    return e, f


def my_while_condition(a=10, b=20):
    return a < b


def function_with_duplicate_output_labels(
    a: int, b: int
) -> tuple[Annotated[int, {"label": "output"}], Annotated[int, {"label": "output"}]]:
    return a + b, a - b


@meta(uri="some URI")
def complex_function(
    x: u(float, units="meter") = 2.0,
    y: u(float, units="second", something_extra=42) = 1,
) -> tuple[
    u(float, units="meter"),
    u(float, units="meter/second", uri="VELOCITY"),
    Annotated[float, {"label": "complex_output"}],
]:
    speed = x / y
    return x, speed, speed / y


@swf.workflow
@meta(uri="some other URI")
def complex_macro(
    x: u(float, units="meter") = 2.0,
):
    a, b, c = complex_function(x)
    return b, c


@swf.workflow
@meta(triples=("a", "b", "c"))
def complex_workflow(
    x: u(float, units="meter") = 2.0,
):
    b, c = complex_macro(x)
    return c


def _serialize_callable_functions(data):
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if key == "function" and callable(value):
                result[key] = flowrep_dict.get_function_metadata(value)
            else:
                result[key] = _serialize_callable_functions(value)
        return result
    if isinstance(data, list):
        return [_serialize_callable_functions(v) for v in data]
    return data


class TestWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None

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
                    "function": {
                        "module": operation.__module__,
                        "qualname": "operation",
                        "version": "not_defined",
                    },
                    "type": "atomic",
                },
                "add_0": {
                    "inputs": {
                        "x": {"dtype": float, "default": 2.0},
                        "y": {"dtype": float, "default": 1},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "function": {
                        "module": add.__module__,
                        "qualname": "add",
                        "version": "not_defined",
                    },
                    "uri": "add",
                    "type": "atomic",
                },
                "multiply_0": {
                    "inputs": {
                        "x": {"dtype": float},
                        "y": {"dtype": float, "default": 5},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "function": {
                        "module": multiply.__module__,
                        "qualname": "multiply",
                        "version": "not_defined",
                    },
                    "type": "atomic",
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
            "type": "workflow",
            "uri": "this macro has metadata",
        }
        wf = example_macro.get_semantikon_dict()
        self.assertEqual(wf["type"], "workflow")
        smtk_wf = _serialize_callable_functions(wf)
        del smtk_wf["function"]
        self.assertEqual(smtk_wf, ref_data)

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
                            "function": {
                                "module": operation.__module__,
                                "qualname": "operation",
                                "version": "not_defined",
                            },
                            "inputs": {"x": {"dtype": float}, "y": {"dtype": float}},
                            "outputs": {
                                "output_0": {"dtype": float},
                                "output_1": {"dtype": float},
                            },
                            "type": "atomic",
                        },
                        "add_0": {
                            "function": {
                                "module": add.__module__,
                                "qualname": "add",
                                "version": "not_defined",
                            },
                            "inputs": {
                                "x": {"dtype": float, "default": 2.0},
                                "y": {"dtype": float, "default": 1},
                            },
                            "outputs": {"output": {"dtype": float}},
                            "uri": "add",
                            "type": "atomic",
                        },
                        "multiply_0": {
                            "function": {
                                "module": multiply.__module__,
                                "qualname": "multiply",
                                "version": "not_defined",
                            },
                            "inputs": {
                                "x": {"dtype": float},
                                "y": {"dtype": float, "default": 5},
                            },
                            "outputs": {"output": {"dtype": float}},
                            "type": "atomic",
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
                    "type": "workflow",
                    "uri": "this macro has metadata",
                },
                "add_0": {
                    "function": {
                        "module": add.__module__,
                        "qualname": "add",
                        "version": "not_defined",
                    },
                    "inputs": {
                        "x": {"dtype": float, "default": 2.0},
                        "y": {"dtype": float, "default": 1},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "uri": "add",
                    "type": "atomic",
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
            "type": "workflow",
        }
        wf = example_workflow.get_semantikon_dict()
        self.assertEqual(wf["type"], "workflow")
        smtk_wf = _serialize_callable_functions(wf)
        del smtk_wf["function"]
        del smtk_wf["nodes"]["example_macro_0"]["function"]
        self.assertEqual(smtk_wf, ref_data)

    def test_forbidden_output_labels(self):
        def test_workflow(a: int, b: int):
            a, b = function_with_duplicate_output_labels(a, b)
            return a, b

        with self.assertRaises(ValueError) as cm:
            swf.workflow(test_workflow)
        self.assertIn(
            "must have unique elements. Duplicates:",
            str(cm.exception),
        )
        self.assertIn("output", str(cm.exception))

        def test_workflow_non_identifier_label(
            a: int, b: int
        ) -> Annotated[float, {"label": "not an identifier"}]:
            result = add(a, b)
            return result

        with self.assertRaises(ValueError) as cm:
            swf.workflow(test_workflow_non_identifier_label)
        self.assertIn("Label must be a valid Python identifier", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
