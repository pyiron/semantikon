"""Tests for the flowrep retrospective → semantikon nested-dict converter."""

import dataclasses
import unittest

from flowrep.api import schemas as frs
from flowrep.api import tools as frt
from rdflib import RDF, RDFS

from semantikon import datastructure, flowrep_dict
from semantikon.metadata import u
from semantikon.ontology import PMD, get_knowledge_graph

# PMD terms probed below; see https://w3id.org/pmd/co
PMD_PROCESS = PMD["0000010"]
PMD_FUNCTION_NAME = PMD["0000100"]


@frt.atomic
def my_add(a, b):
    return a + b


def my_mul(a, b):
    return a * b


@frt.atomic
def negate(x):
    return -x


# Reuse the diamond recipe from the test suite — it has a reference, so we get
# annotations and defaults on the live Workflow.
@frt.workflow
def _diamond_workflow(a: int, b: int = 1) -> int:
    s = my_add(a, b)
    n = negate(a)
    result = my_mul(s, n)
    return result


@frt.workflow
def _passthrough_workflow(x):
    y = negate(x)
    return x, y


@frt.atomic
def increment(x, step=1):
    return x + step


@frt.atomic
def divmod_func(a: float, b: float) -> tuple[float, float]:
    quotient = a // b
    remainder = a % b
    return quotient, remainder


def operation(x: float, y: float) -> tuple[float, float]:
    return x + y, x - y


def add(x: float = 2.0, y: float = 1) -> float:
    return x + y


def multiply(x: float, y: float = 5) -> float:
    return x * y


@frt.workflow
def workflow_with_data(a=10, b=20):
    x = add(a, b)
    y = multiply(x, b)
    return x, y


@frt.workflow
def example_macro(a=10, b=20):
    c, d = operation(a, b)
    e = add(c, y=d)
    f = multiply(e)
    return f


@frt.workflow
def example_workflow(a=10, b=20):
    y = example_macro(a, b)
    z = add(y, b)
    return z


@dataclasses.dataclass
class TestClass:
    a: int = 10
    b: int = 20


def some_function(test: TestClass):
    return test


@frt.workflow
def workflow_with_class(test: TestClass):
    test = some_function(test)
    return test


def example_function(x):
    """An example function to be hashed."""
    return x * 2


@frt.atomic
def identity(x):
    return x


@frt.atomic
def measure(
    distance: u(float, units="meter", uri="http://example.org/distance"),
) -> u(float, units="meter", label="measured"):
    return distance


@frt.workflow
def measurement_workflow(
    distance: u(float, units="meter", uri="http://example.org/distance"),
) -> u(float, units="meter", label="velocity"):
    result = measure(distance)
    return result


@frt.workflow
def _passthrough_with_child(x):
    _ = identity(x)
    return x


_passthrough_with_child.flowrep_recipe = frs.WorkflowRecipe(
    inputs=["x"],
    outputs=["output_0"],
    nodes={"identity_0": identity.flowrep_recipe},
    input_edges={
        frs.TargetHandle(node="identity_0", port="x"): frs.InputSource(port="x"),
    },
    edges={},
    output_edges={
        frs.OutputTarget(port="output_0"): frs.InputSource(port="x"),
    },
    reference=_passthrough_with_child.flowrep_recipe.reference,
)


def _inner_workflow_with_output_0() -> frs.WorkflowRecipe:
    """A workflow whose sole output is named ``output_0``.

    This is unusual (the workflow parser names outputs from return variables),
    but perfectly valid for manually constructed recipes.
    """
    return frs.WorkflowRecipe(
        inputs=["x"],
        outputs=["output_0"],
        nodes={"identity_0": identity.flowrep_recipe},
        input_edges={
            frs.TargetHandle(node="identity_0", port="x"): frs.InputSource(port="x"),
        },
        edges={},
        output_edges={
            frs.OutputTarget(port="output_0"): frs.SourceHandle(
                node="identity_0", port="x"
            ),
        },
    )


def _parent_workflow(inner: frs.WorkflowRecipe) -> frs.WorkflowRecipe:
    """A workflow that wraps *inner* and reads its ``output_0`` port."""
    return frs.WorkflowRecipe(
        inputs=["x"],
        outputs=["result"],
        nodes={"inner_0": inner},
        input_edges={
            frs.TargetHandle(node="inner_0", port="x"): frs.InputSource(port="x"),
        },
        edges={},
        output_edges={
            frs.OutputTarget(port="result"): frs.SourceHandle(
                node="inner_0", port="output_0"
            ),
        },
    )


class TestNonNodeToDict(unittest.TestCase):
    def test_non_node_type_raises(self):
        with self.assertRaisesRegex(TypeError, "Unsupported data node type"):
            flowrep_dict.nodedata2dict(123)


class TestAtomicToDict(unittest.TestCase):
    def test_basic_structure(self):
        node = frs.AtomicData.from_recipe(my_add.flowrep_recipe)
        d = flowrep_dict.nodedata2dict(node)
        self.assertEqual(d["type"], "atomic")
        self.assertIn("function", d)
        self.assertTrue(callable(d["function"]))
        self.assertIn("inputs", d)
        self.assertIn("outputs", d)

    def test_pre_run(self):
        node = frs.AtomicData.from_recipe(my_add.flowrep_recipe)
        d = flowrep_dict.nodedata2dict(node)
        self.assertIn("inputs", d)
        self.assertIn("outputs", d)
        self.assertIn("a", d["inputs"])
        self.assertIn("b", d["inputs"])
        # Pre-run: no values populated
        self.assertNotIn("value", d["inputs"]["a"])
        self.assertNotIn("value", d["outputs"]["output"])

    def test_post_run(self):
        node = frt.run_recipe(my_add.flowrep_recipe, a=3, b=4)
        d = flowrep_dict.nodedata2dict(node)
        self.assertEqual(d["inputs"]["a"]["value"], 3)
        self.assertEqual(d["inputs"]["b"]["value"], 4)
        self.assertEqual(d["outputs"]["output"]["value"], 7)

    def test_defaults_included(self):
        node = frs.AtomicData.from_recipe(increment.flowrep_recipe)
        d = flowrep_dict.nodedata2dict(node)
        self.assertEqual(d["inputs"]["step"]["default"], 1)
        self.assertNotIn("default", d["inputs"]["x"])

    def test_multi_output(self):
        node = frt.run_recipe(divmod_func.flowrep_recipe, a=17, b=5)
        d = flowrep_dict.nodedata2dict(node)
        self.assertIn("quotient", d["outputs"])
        self.assertIn("remainder", d["outputs"])
        self.assertAlmostEqual(d["outputs"]["quotient"]["value"], 3.0)
        self.assertAlmostEqual(d["outputs"]["remainder"]["value"], 2.0)


class TestWorkflowToDict(unittest.TestCase):
    def test_basic_structure(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = frs.DagData.from_recipe(recipe)
        d = flowrep_dict.nodedata2dict(node)
        self.assertEqual(d["type"], "workflow")
        self.assertTrue(callable(d["function"]))
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        self.assertIn("label", d)
        self.assertIn("inputs", d)
        self.assertIn("outputs", d)
        # child functions available recursively
        self.assertTrue(callable(d["nodes"]["my_add_0"]["function"]))

    def test_label_inferred_from_reference(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = frs.DagData.from_recipe(recipe)
        d = flowrep_dict.nodedata2dict(node)
        self.assertEqual(d["label"], "_diamond_workflow")

    def test_label_override(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = frs.DagData.from_recipe(recipe)
        d = flowrep_dict.nodedata2dict(node, label="my_label")
        self.assertEqual(d["label"], "my_label")

    def test_child_nodes_present(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = frs.DagData.from_recipe(recipe)
        d = flowrep_dict.nodedata2dict(node)
        self.assertIn("my_add_0", d["nodes"])
        self.assertIn("negate_0", d["nodes"])
        self.assertIn("my_mul_0", d["nodes"])
        for child_d in d["nodes"].values():
            self.assertEqual(child_d["type"], "atomic")

    def test_edges_cover_all_recipe_edges(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = frs.DagData.from_recipe(recipe)
        d = flowrep_dict.nodedata2dict(node)
        edges = d["edges"]
        # Should have input_edges + sibling edges + output_edges
        n_expected = (
            len(recipe.input_edges) + len(recipe.edges) + len(recipe.output_edges)
        )
        self.assertEqual(len(edges), n_expected)

    def test_passthrough_edges(self):
        recipe = _passthrough_workflow.flowrep_recipe
        node = frs.DagData.from_recipe(recipe)
        d = flowrep_dict.nodedata2dict(node)
        self.assertIn(("inputs.x", "outputs.x"), d["edges"])

    def test_edge_format(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = frs.DagData.from_recipe(recipe)
        d = flowrep_dict.nodedata2dict(node)
        for src, tgt in d["edges"]:
            self.assertIsInstance(src, str)
            self.assertIsInstance(tgt, str)
            # Every edge string should contain at least one dot
            self.assertIn(".", src)
            self.assertIn(".", tgt)

    def test_pre_run(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = frs.DagData.from_recipe(recipe)
        d = flowrep_dict.nodedata2dict(node)
        self.assertIn("inputs", d)
        self.assertIn("outputs", d)
        self.assertIn("a", d["inputs"])
        self.assertIn("b", d["inputs"])
        # Default from the reference signature
        self.assertEqual(d["inputs"]["b"]["default"], 1)
        self.assertIn("result", d["outputs"])

    def test_post_run(self):
        wf = frt.run_recipe(_diamond_workflow.flowrep_recipe, a=3, b=7)
        d = flowrep_dict.nodedata2dict(wf)
        self.assertEqual(d["inputs"]["a"]["value"], 3)
        self.assertEqual(d["inputs"]["b"]["value"], 7)
        self.assertEqual(d["outputs"]["result"]["value"], (3 + 7) * (-3))

    def test_child_io_post_run(self):
        """After execution, child nodes also carry values."""
        wf = frt.run_recipe(_diamond_workflow.flowrep_recipe, a=3, b=7)
        d = flowrep_dict.nodedata2dict(wf)
        add_d = d["nodes"]["my_add_0"]
        self.assertEqual(add_d["inputs"]["a"]["value"], 3)
        self.assertEqual(add_d["inputs"]["b"]["value"], 7)
        self.assertEqual(add_d["outputs"]["output"]["value"], 10)

    def test_edges_self_consistent(self):
        """Every port referenced in an edge should correspond to a real node or
        the workflow's own inputs/outputs."""
        recipe = _diamond_workflow.flowrep_recipe
        node = frs.DagData.from_recipe(recipe)
        d = flowrep_dict.nodedata2dict(node)

        valid_prefixes = {"inputs", "outputs"} | set(d["nodes"].keys())
        for src, tgt in d["edges"]:
            src_prefix = src.split(".")[0]
            tgt_prefix = tgt.split(".")[0]
            self.assertIn(src_prefix, valid_prefixes, msg=f"bad edge source: {src}")
            self.assertIn(tgt_prefix, valid_prefixes, msg=f"bad edge target: {tgt}")

    def test_dict_to_nodedata_round_trip(self):
        wf_dict = flowrep_dict.nodedata2dict(
            frt.run_recipe(_diamond_workflow.flowrep_recipe, a=3, b=7)
        )
        node = flowrep_dict.dict_to_nodedata(wf_dict)
        self.assertIsInstance(node, frs.DagData)
        self.assertEqual(node.input_ports["a"].value, 3)
        self.assertEqual(node.input_ports["b"].value, 7)
        self.assertEqual(node.nodes["my_add_0"].output_ports["output"].value, 10)
        self.assertEqual(
            flowrep_dict.nodedata2dict(node)["edges"],
            wf_dict["edges"],
        )


class TestFunctionlessWorkflow(unittest.TestCase):
    """A workflow dict without a ``function`` key (e.g. a pyiron_workflow generic
    ``Workflow``) is a reference-free flowrep workflow: IO is taken from the dict."""

    def _wf_dict(self):
        return {
            "type": "workflow",
            "label": "manual_wf",
            "inputs": {"a": {}, "b": {}},
            "outputs": {"total": {}},
            "nodes": {
                "add": {
                    "type": "atomic",
                    "function": my_add,
                    "inputs": {"a": {}, "b": {}},
                    "outputs": {"output": {}},
                },
            },
            "edges": [
                ("inputs.a", "add.inputs.a"),
                ("inputs.b", "add.inputs.b"),
                ("add.outputs.output", "outputs.total"),
            ],
        }

    def test_recipe_is_reference_free(self):
        recipe = flowrep_dict._dict_to_workflow_recipe(self._wf_dict())
        self.assertIsInstance(recipe, frs.WorkflowRecipe)
        self.assertIsNone(recipe.reference)
        self.assertEqual(list(recipe.inputs), ["a", "b"])
        self.assertEqual(list(recipe.outputs), ["total"])

    def test_dict_to_nodedata(self):
        node = flowrep_dict.dict_to_nodedata(self._wf_dict())
        self.assertIsInstance(node, frs.DagData)
        self.assertEqual(list(node.input_ports), ["a", "b"])
        self.assertEqual(list(node.output_ports), ["total"])

    def test_knowledge_graph_is_populated(self):
        """The reference-free workflow still yields a real, non-empty graph: its
        own IO ports and its child atomic's process are all represented."""
        g = get_knowledge_graph(flowrep_dict._dict_to_workflow_recipe(self._wf_dict()))
        self.assertGreater(len(g), 0)
        labels = {str(o) for _, p, o in g if p == RDFS.label}
        # the child atomic, carrying its function semantics, is in the graph
        self.assertIn("Function name 'my_add'", labels)

    def test_knowledge_graph_omits_workflow_process(self):
        """ "Simple": a reference-free workflow contributes structure but no process
        of its own. Only the child atomic appears as a process/function -- a
        function-bearing workflow would additionally represent the workflow itself
        (e.g. the diamond workflow yields four such nodes, one per child plus root)."""
        g = get_knowledge_graph(flowrep_dict._dict_to_workflow_recipe(self._wf_dict()))
        processes = [s for s, p, o in g if p == RDF.type and o == PMD_PROCESS]
        function_names = [
            s for s, p, o in g if p == RDF.type and o == PMD_FUNCTION_NAME
        ]
        self.assertEqual(len(processes), 1, "only the child atomic is a process")
        self.assertEqual(
            len(function_names), 1, "the workflow root carries no function"
        )


class TestFlowControlStub(unittest.TestCase):
    def test_raises_not_implemented(self):
        recipe = frs.ForEachRecipe(
            inputs=["xs"],
            outputs=["ys"],
            body_node=frs.LabeledRecipe(label="body", recipe=negate.flowrep_recipe),
            input_edges={
                frs.TargetHandle(node="body", port="x"): frs.InputSource(port="xs")
            },
            output_edges={
                frs.OutputTarget(port="ys"): frs.SourceHandle(
                    node="body", port="output_0"
                )
            },
            nested_ports=["x"],
        )
        fc = frs.FlowControlData.from_recipe(recipe)
        with self.assertRaises(NotImplementedError):
            flowrep_dict.nodedata2dict(fc)


class TestRoundTripConsistency(unittest.TestCase):
    """Verify that pre-run and post-run dicts are structurally compatible."""

    def test_pre_and_post_run_same_keys(self):
        recipe = _diamond_workflow.flowrep_recipe
        pre = flowrep_dict.nodedata2dict(frs.DagData.from_recipe(recipe))
        post = flowrep_dict.nodedata2dict(frt.run_recipe(recipe, a=3, b=7))
        # Same top-level keys
        self.assertEqual(set(pre.keys()), set(post.keys()))
        # Same node labels
        self.assertEqual(set(pre["nodes"].keys()), set(post["nodes"].keys()))
        # Same edges (edges come from the recipe, not execution)
        self.assertEqual(sorted(pre["edges"]), sorted(post["edges"]))

    def test_pre_run_no_values(self):
        recipe = _diamond_workflow.flowrep_recipe
        d = flowrep_dict.nodedata2dict(frs.DagData.from_recipe(recipe))
        for port_d in d["outputs"].values():
            self.assertNotIn("value", port_d)

    def test_post_run_has_values(self):
        d = flowrep_dict.nodedata2dict(
            frt.run_recipe(_diamond_workflow.flowrep_recipe, a=3, b=7),
        )
        for port_d in d["outputs"].values():
            self.assertIn("value", port_d)


class TestAnnotationConverters(unittest.TestCase):
    def test_type_hint_none(self):
        self.assertIsNone(flowrep_dict.annotation_to_type_hint(None))

    def test_type_hint_plain(self):
        self.assertIs(flowrep_dict.annotation_to_type_hint(float), float)

    def test_type_hint_strips_annotated_to_base(self):
        self.assertIs(flowrep_dict.annotation_to_type_hint(u(int, units="meter")), int)

    def test_type_metadata_none(self):
        self.assertIsNone(flowrep_dict.annotation_to_type_metadata(None))

    def test_type_metadata_plain_is_none(self):
        self.assertIsNone(flowrep_dict.annotation_to_type_metadata(float))

    def test_type_metadata_from_annotated(self):
        meta = flowrep_dict.annotation_to_type_metadata(u(int, units="meter"))
        self.assertIsInstance(meta, datastructure.TypeMetadata)
        self.assertEqual(meta.units, "meter")

    def test_port_dict_strips_annotated_and_propagates_metadata(self):
        """_port_dict stores the bare dtype and flattens semantikon metadata."""
        d = flowrep_dict._port_dict(42, u(int, units="meter"))
        self.assertIs(d["dtype"], int)
        self.assertEqual(d["value"], 42)
        self.assertEqual(d["units"], "meter")


class TestDigraphConverters(unittest.TestCase):
    def test_wf_dict_to_graph(self):
        wf_dict = flowrep_dict.nodedata2dict(
            frt.run_recipe(example_workflow.flowrep_recipe, a=1, b=2)
        )
        G = flowrep_dict._get_workflow_graph(wf_dict)
        self.assertDictEqual(
            G.nodes["add_0:inputs@y"],
            {"value": 2, "dtype": float, "default": 1, "position": 1, "step": "input"},
        )
        self.assertDictEqual(G.nodes["outputs@z"], {"step": "output", "value": 12})

    def test_get_hashed_node_dict(self):

        # workflow_dict = workflow_with_data.run(a=10, b=20)
        workflow_dict = flowrep_dict.nodedata2dict(
            frt.run_recipe(workflow_with_data.flowrep_recipe, a=10, b=20),
        )
        hashed_dict = flowrep_dict.get_hashed_node_dict(workflow_dict)
        for node in hashed_dict.values():
            self.assertIn("hash", node)
            self.assertIsInstance(node["hash"], str)
            self.assertEqual(len(node["hash"]), 64)
        self.assertTrue(
            hashed_dict["multiply_0"]["inputs"]["x"].endswith(
                hashed_dict["add_0"]["hash"] + "@output"
            )
        )

        # workflow_dict = workflow_with_data.get_flowrep_dict()
        workflow_dict = flowrep_dict.nodedata2dict(
            frt.recipe2data(workflow_with_data.flowrep_recipe)
        )
        hashed_dict = flowrep_dict.get_hashed_node_dict(workflow_dict)
        for node in hashed_dict.values():
            self.assertNotIn("hash", node)
        workflow_dict["inputs"] = {"a": {"value": 10}, "b": {"value": 20}}
        # workflow_dict_run = workflow_with_data.run(a=10, b=20)
        workflow_dict_run = flowrep_dict.nodedata2dict(
            frt.run_recipe(workflow_with_data.flowrep_recipe, a=10, b=20)
        )
        self.assertDictEqual(
            flowrep_dict.get_hashed_node_dict(workflow_dict),
            flowrep_dict.get_hashed_node_dict(workflow_dict_run),
        )

        # workflow_dict = example_workflow.run(a=10, b=20)
        workflow_dict = flowrep_dict.nodedata2dict(
            frt.run_recipe(example_workflow.flowrep_recipe, a=10, b=20)
        )
        hashed_dict = flowrep_dict.get_hashed_node_dict(workflow_dict)
        self.assertIn("example_macro_0.operation_0", hashed_dict)

        test_instance = TestClass()
        # workflow_dict = workflow_with_class.run(test=test_instance)
        workflow_dict = flowrep_dict.nodedata2dict(
            frt.run_recipe(workflow_with_class.flowrep_recipe, test=test_instance),
        )
        hashed_dict = flowrep_dict.get_hashed_node_dict(workflow_dict)
        for node in hashed_dict.values():
            self.assertIn("hash", node)
            self.assertIsInstance(node["hash"], str)
            self.assertEqual(len(node["hash"]), 64)


class TestTools(unittest.TestCase):
    def test_get_function_metadata(self):
        meta = flowrep_dict.get_function_metadata(example_function, full_metadata=True)
        self.assertIn("name", meta)
        self.assertIn("module", meta)
        self.assertIn("docstring", meta)
        self.assertEqual(meta["name"], "example_function")
        meta = flowrep_dict.get_function_metadata(example_function, full_metadata=False)
        self.assertNotIn("docstring", meta)


class TestOutputSanitizationConsistency(unittest.TestCase):
    """Edges and output dicts must agree on port names after sanitization."""

    def test_wfms_runs_fine(self):
        """The recipe itself is well-formed and executes correctly."""
        inner = _inner_workflow_with_output_0()
        parent = _parent_workflow(inner)
        node = frt.run_recipe(parent, x=42)
        self.assertEqual(node.output_ports["result"].value, 42)

    def test_inner_dict_edges_consistent_with_outputs(self):
        """The inner workflow's edges and outputs dict must use the same name."""
        inner = _inner_workflow_with_output_0()
        inner_data = frs.DagData.from_recipe(inner)
        d = flowrep_dict.nodedata2dict(inner_data)

        # The outputs dict uses "output"
        self.assertIn("output", d["outputs"])
        self.assertNotIn("output_0", d["outputs"])

        # Edges must also reference "outputs.output", not "outputs.output_0"
        output_edge_targets = [
            tgt for _, tgt in d["edges"] if tgt.startswith("outputs.")
        ]
        self.assertIn("outputs.output", output_edge_targets)
        self.assertNotIn("outputs.output_0", output_edge_targets)

    def test_nested_dict_produces_single_inner_output_node(self):
        """get_workflow_graph must create exactly one output node per port."""
        inner = _inner_workflow_with_output_0()
        parent = _parent_workflow(inner)
        parent_data = frs.DagData.from_recipe(parent)
        d = flowrep_dict.nodedata2dict(parent_data)
        G = flowrep_dict._get_workflow_graph(d)

        inner_output_nodes = [n for n in G.nodes if n.startswith("inner_0:outputs@")]
        output_names = [n.split("@")[-1] for n in inner_output_nodes]
        self.assertEqual(
            output_names,
            ["output"],
            msg="Should be exactly one inner output node named 'output'",
        )

    def test_nested_dict_run_produces_right_value(self):
        """The full recipe run path must produce the right value."""
        inner = _inner_workflow_with_output_0()
        parent = _parent_workflow(inner)
        parent_data = frt.run_recipe(parent, x=42)
        d = flowrep_dict.nodedata2dict(parent_data)
        self.assertEqual(d["outputs"]["result"]["value"], 42)

    def test_dict_to_nodedata_normalizes_passthrough_output_name(self):
        node = frs.DagData.from_recipe(_passthrough_with_child.flowrep_recipe)
        d = flowrep_dict.nodedata2dict(node)
        self.assertIn(("inputs.x", "outputs.output"), d["edges"])
        round_tripped = flowrep_dict.dict_to_nodedata(d)
        self.assertIn(
            frs.OutputTarget(port="output_0"),
            round_tripped.recipe.output_edges,
        )
        self.assertEqual(
            round_tripped.recipe.output_edges[frs.OutputTarget(port="output_0")],
            frs.InputSource(port="x"),
        )


class TestGetFunctionMetadata(unittest.TestCase):
    def test_getter_gets_already_gotten(self):
        """Existing tests miss one of the hinted cases, so test it explicitly"""
        sure_looks_like_metadata = {
            "module": "foo",
            "qualname": "bar",
            "unstructured_data_is_risky": "because this is not a field we wanted",
        }
        reparsed = flowrep_dict.get_function_metadata(sure_looks_like_metadata)
        self.assertIs(
            reparsed,
            sure_looks_like_metadata,
            msg="if it already looks like metadata, the code has a clause to return "
            "that",
        )
        self.assertNotIn(
            "hash",
            reparsed,
            msg="that is a dangerous move, since we asked for full metadata but "
            "short-circuited finding the hash by using an early return! It would "
            "probably be nice if this test failed.",
        )

    def test_non_metadata_dict_raises(self):
        with self.assertRaisesRegex(ValueError, "it doesn't look like metadata"):
            flowrep_dict.get_function_metadata({"foo": "bar"})


class TestMetadataPropagation(unittest.TestCase):
    """``semantikon.u`` metadata must survive the dataclass → dict conversion.

    Exercises the whole path: a ``u``-wrapped workflow signature → flowrep
    recipe → :class:`frs.DagData` → :func:`node_data_to_dict`, asserting the
    :class:`~semantikon.datastructure.TypeMetadata` fields land in both the
    workflow-level and the nested child-node port dicts.
    """

    def setUp(self):
        node = frs.DagData.from_recipe(measurement_workflow.flowrep_recipe)
        self.d = flowrep_dict.nodedata2dict(node)

    def test_workflow_input_metadata(self):
        port = self.d["inputs"]["distance"]
        self.assertIs(port["dtype"], float)
        self.assertEqual(port["units"], "meter")
        self.assertEqual(port["uri"], "http://example.org/distance")

    def test_workflow_output_metadata(self):
        port = self.d["outputs"]["result"]
        self.assertIs(port["dtype"], float)
        self.assertEqual(port["units"], "meter")
        self.assertEqual(port["label"], "velocity")

    def test_child_node_input_metadata(self):
        port = self.d["nodes"]["measure_0"]["inputs"]["distance"]
        self.assertIs(port["dtype"], float)
        self.assertEqual(port["units"], "meter")
        self.assertEqual(port["uri"], "http://example.org/distance")

    def test_child_node_output_metadata(self):
        # ``measure`` returns the variable ``distance``, so the output port
        # keeps that name rather than being sanitized to "output".
        port = self.d["nodes"]["measure_0"]["outputs"]["distance"]
        self.assertIs(port["dtype"], float)
        self.assertEqual(port["units"], "meter")
        self.assertEqual(port["label"], "measured")


if __name__ == "__main__":
    unittest.main()
