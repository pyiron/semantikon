import unittest

import networkx as nx
from flowrep import workflow as fwf, tools

import semantikon.workflow as swf
from semantikon import datastructure
from semantikon.converter import parse_input_args
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


@meta(uri="some URI")
def complex_function(
    x: u(float, units="meter") = 2.0,
    y: u(float, units="second", something_extra=42) = 1,
) -> tuple[
    u(float, units="meter"),
    u(float, units="meter/second", uri="VELOCITY"),
    float,
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


class TestWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None

    def test_get_node_dict(self):
        node_dict = swf._get_node_dict(add)
        self.assertEqual(
            node_dict,
            {
                "inputs": {
                    "x": {"dtype": float, "default": 2.0},
                    "y": {"dtype": float, "default": 1},
                },
                "outputs": {"output": {"dtype": float}},
                "function": add,
                "uri": "add",
                "type": "Function",
            },
        )

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
                        "module": "__main__",
                        "qualname": "operation",
                        "version": "not_defined",
                    },
                    "type": "Function",
                },
                "add_0": {
                    "inputs": {
                        "x": {"dtype": float, "default": 2.0},
                        "y": {"dtype": float, "default": 1},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "function": {
                        "module": "__main__",
                        "qualname": "add",
                        "version": "not_defined",
                    },
                    "uri": "add",
                    "type": "Function",
                },
                "multiply_0": {
                    "inputs": {
                        "x": {"dtype": float},
                        "y": {"dtype": float, "default": 5},
                    },
                    "outputs": {"output": {"dtype": float}},
                    "function": {
                        "module": "__main__",
                        "qualname": "multiply",
                        "version": "not_defined",
                    },
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
            "uri": "this macro has metadata",
        }
        wf = example_macro.serialize_workflow()
        self.assertEqual(wf["type"], "Workflow")
        smtk_wf = tools.serialize_functions(wf)
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
                                "module": "__main__",
                                "qualname": "operation",
                                "version": "not_defined",
                            },
                            "inputs": {"x": {"dtype": float}, "y": {"dtype": float}},
                            "outputs": {
                                "output_0": {"dtype": float},
                                "output_1": {"dtype": float},
                            },
                            "type": "Function",
                        },
                        "add_0": {
                            "function": {
                                "module": "__main__",
                                "qualname": "add",
                                "version": "not_defined",
                            },
                            "inputs": {
                                "x": {"dtype": float, "default": 2.0},
                                "y": {"dtype": float, "default": 1},
                            },
                            "outputs": {"output": {"dtype": float}},
                            "uri": "add",
                            "type": "Function",
                        },
                        "multiply_0": {
                            "function": {
                                "module": "__main__",
                                "qualname": "multiply",
                                "version": "not_defined",
                            },
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
                    "function": {
                        "module": "__main__",
                        "qualname": "add",
                        "version": "not_defined",
                    },
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
        wf = example_workflow.serialize_workflow()
        self.assertEqual(wf["type"], "Workflow")
        smtk_wf = tools.serialize_functions(wf)
        del smtk_wf["function"]
        del smtk_wf["nodes"]["example_macro_0"]["function"]
        self.assertEqual(smtk_wf, ref_data)

    def test_separate_types(self):
        old_data = swf.get_workflow_dict(example_workflow)
        class_dict = swf.separate_types(old_data)[1]
        self.assertEqual(class_dict, {"float": float})

    def test_get_node_outputs(self):
        self.assertEqual(
            swf._get_node_outputs(operation, counts=2),
            {"output_0": {"dtype": float}, "output_1": {"dtype": float}},
        )
        self.assertEqual(
            swf._get_node_outputs(operation, counts=1),
            {"output": {"dtype": tuple[float, float]}},
        )
        self.assertEqual(
            swf._get_node_outputs(parallel_execution, counts=2),
            {"e": {}, "f": {}},
        )
        self.assertEqual(
            swf._get_node_outputs(parallel_execution, counts=1),
            {"output": {}},
        )

    def test_get_workflow_output(self):

        def test_function_1(a, b):
            return a + b

        self.assertEqual(
            swf._get_node_outputs(test_function_1),
            {"output": {}},
        )

        def test_function_2(a, b):
            return a

        self.assertEqual(
            swf._get_node_outputs(test_function_2),
            {"a": {}},
        )

        def test_function_3(a, b):
            return a, b

        self.assertEqual(
            swf._get_node_outputs(test_function_3),
            {"a": {}, "b": {}},
        )

        def test_function_4(a, b):
            return a + b, b

        data = swf._get_node_outputs(test_function_4)
        self.assertEqual(data, {"output_0": {}, "b": {}})
        data["output_0"]["value"] = 0
        self.assertEqual(
            data,
            {"output_0": {"value": 0}, "b": {}},
        )

        def test_function_5(a: int, b: int) -> tuple[int, int]:
            return a, b

        self.assertEqual(
            swf._get_node_outputs(test_function_5),
            {"a": {"dtype": int}, "b": {"dtype": int}},
        )

    def test_ports(self):
        for fnc in (operation, add, multiply, my_while_condition, complex_function):
            with self.subTest(fnc=fnc, msg=fnc.__name__):
                inputs, outputs = swf.get_ports(fnc)
                full_entry = swf._get_node_dict(fnc)
                for entry, node in (
                    (full_entry["inputs"], inputs),
                    (full_entry["outputs"], outputs),
                ):
                    with self.subTest(node.__class__.__name__):
                        node_dictionary = node.to_dictionary()

                        # Transform the node to match the existing style
                        for arg_dictionary in node_dictionary.values():
                            arg_dictionary.pop("label")
                            arg_dictionary.pop("type")
                            metadata = arg_dictionary.pop("metadata", {})
                            arg_dictionary.update(metadata)  # Flatten the metadata

                        self.assertDictEqual(
                            entry,
                            node_dictionary,
                            msg="Dictionary representation must be equivalent to "
                            "existing dictionaries",
                        )

    def test_complex_function_node(self):
        node = swf.get_node(complex_function)

        with self.subTest("Node parsing"):
            self.assertIsInstance(node, swf.Function)
            self.assertIsInstance(node.inputs, swf.Inputs)
            self.assertIsInstance(node.outputs, swf.Outputs)
            self.assertIsInstance(node.metadata, swf.CoreMetadata)
            self.assertEqual(node.type, datastructure.Function.__name__)
            self.assertEqual(node.label, complex_function.__name__)
            self.assertEqual(node.metadata.uri, "some URI")

        with self.subTest("Input parsing"):
            self.assertIsInstance(node.inputs.x, swf.Input)
            self.assertIs(node.inputs.x.dtype, float)
            self.assertAlmostEqual(node.inputs.x.default, 2.0)
            self.assertIsInstance(node.inputs.x.metadata, datastructure.TypeMetadata)
            self.assertEqual(node.inputs.x.metadata.units, "meter")
            self.assertIs(node.inputs.y.dtype, float)
            self.assertAlmostEqual(node.inputs.y.default, 1.0)
            self.assertIsInstance(node.inputs.y.metadata, datastructure.TypeMetadata)
            self.assertEqual(node.inputs.y.metadata.units, "second")
            self.assertEqual(node.inputs.y.metadata.extra["something_extra"], 42)

        with self.subTest("Output parsing"):
            self.assertIsInstance(node.outputs.x, swf.Output)
            self.assertIs(node.outputs.x.dtype, float)
            self.assertIsInstance(node.outputs.x.metadata, datastructure.TypeMetadata)
            self.assertEqual(node.outputs.x.metadata.units, "meter")
            self.assertIs(node.outputs.speed.dtype, float)
            self.assertIsInstance(
                node.outputs.speed.metadata, datastructure.TypeMetadata
            )
            self.assertEqual(node.outputs.speed.metadata.units, "meter/second")
            self.assertEqual(node.outputs.speed.metadata.uri, "VELOCITY")
            self.assertIs(node.outputs.output_2.dtype, float)

    def test_complex_macro(self):
        node = swf.get_node(complex_macro)
        with self.subTest("Node parsing"):
            self.assertIsInstance(node, swf.Workflow)
            self.assertIsInstance(node.inputs, swf.Inputs)
            self.assertAlmostEqual(node.inputs.x.default, 2.0)
            self.assertIsInstance(node.inputs.x.metadata, datastructure.TypeMetadata)
            self.assertEqual(node.inputs.x.metadata.units, "meter")
            self.assertIsInstance(node.outputs, swf.Outputs)
            self.assertIsInstance(node.metadata, swf.CoreMetadata)
            self.assertEqual(node.type, datastructure.Workflow.__name__)
            self.assertEqual(node.label, complex_macro.__name__)
            self.assertEqual(node.metadata.uri, "some other URI")

        with self.subTest("Graph-node parsing"):
            self.assertIsInstance(node.nodes, swf.Nodes)
            self.assertIsInstance(node.nodes.complex_function_0, swf.Function)
            self.assertEqual("complex_function_0", node.nodes.complex_function_0.label)
            self.assertIsInstance(node.edges, swf.Edges)
            self.assertDictEqual(
                {
                    "complex_function_0.inputs.x": "inputs.x",
                    "outputs.b": "complex_function_0.outputs.1",
                    "outputs.c": "complex_function_0.outputs.2",
                },
                node.edges.to_dictionary(),
            )

    def test_complex_workflow(self):
        node = swf.get_node(complex_workflow)
        with self.subTest("Node parsing"):
            self.assertIsInstance(node, swf.Workflow)
            self.assertIsInstance(node.inputs, swf.Inputs)
            self.assertAlmostEqual(node.inputs.x.default, 2.0)
            self.assertIsInstance(node.inputs.x.metadata, datastructure.TypeMetadata)
            self.assertEqual(node.inputs.x.metadata.units, "meter")
            self.assertIsInstance(node.outputs, swf.Outputs)
            self.assertIsInstance(node.metadata, swf.CoreMetadata)
            self.assertEqual(node.type, datastructure.Workflow.__name__)
            self.assertEqual(node.label, complex_workflow.__name__)
            self.assertTupleEqual(node.metadata.triples, ("a", "b", "c"))

        with self.subTest("Graph-node parsing"):
            self.assertIsInstance(node.nodes, swf.Nodes)
            self.assertIsInstance(node.nodes.complex_macro_0, swf.Workflow)
            self.assertIsInstance(node.edges, swf.Edges)
            self.assertDictEqual(
                {
                    "complex_macro_0.inputs.x": "inputs.x",
                    "outputs.c": "complex_macro_0.outputs.c",
                },
                node.edges.to_dictionary(),
            )

    def test_function(self):
        for fnc in (operation, add, multiply, my_while_condition):
            with self.subTest(fnc=fnc, msg=fnc.__name__):
                entry = swf._get_node_dict(
                    fnc,
                    parse_input_args(fnc),
                    swf._get_node_outputs(fnc),
                )
                # Cheat and modify the entry to resemble the node structure
                if hasattr(fnc, "_semantikon_metadata"):
                    # Nest the metadata in the entry
                    metadata = fnc._semantikon_metadata
                    for k in metadata.keys():
                        entry.pop(k)
                    entry["metadata"] = metadata

                node_dictionary = swf.get_node(fnc).to_dictionary()
                # Cheat and modify the node_dictionary to match the entry format
                node_dictionary.pop("label")
                for io in (node_dictionary["inputs"], node_dictionary["outputs"]):
                    for port_dictionary in io.values():
                        port_dictionary.pop("type")
                        port_dictionary.pop("label")

                self.assertDictEqual(
                    entry,
                    node_dictionary,
                    msg="Just an interim cyclicity test",
                )

    def test_edges_to_output_counts(self):
        self.assertDictEqual(
            swf._edges_to_output_counts(
                example_macro.serialize_workflow()["edges"],
            ),
            {'operation_0': 2, 'add_0': 1, 'multiply_0': 1}
        )

if __name__ == "__main__":
    unittest.main()
