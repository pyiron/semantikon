"""Tests for the live → nested-dict converter."""

import unittest

from flowrep.models import live, wfms
from flowrep.models.parsers import workflow_parser

from flowrep_static import library


from semantikon import flowrep_dict


def my_add(a, b):
    return a + b


def my_mul(a, b):
    return a * b


def negate(x):
    return -x


# Reuse the diamond recipe from the test suite — it has a reference, so we get
# annotations and defaults on the live Workflow.
@workflow_parser.workflow
def _diamond_workflow(a: int, b: int = 1) -> int:
    s = library.my_add(a, b)
    n = library.negate(a)
    result = library.my_mul(s, n)
    return result


class TestAtomicToDict(unittest.TestCase):
    def test_basic_structure(self):
        node = live.Atomic.from_recipe(library.my_add.flowrep_recipe)
        d = flowrep_dict.live_to_dict(node)
        self.assertEqual(d["type"], "atomic")
        self.assertIn("function", d)
        # Default: metadata dict, not raw callable
        self.assertIsInstance(d["function"], dict)
        self.assertIn("module", d["function"])
        self.assertNotIn("inputs", d)
        self.assertNotIn("outputs", d)

    def test_with_function(self):
        node = live.Atomic.from_recipe(library.my_add.flowrep_recipe)
        d = flowrep_dict.live_to_dict(node, with_function=True)
        self.assertTrue(callable(d["function"]))

    def test_with_io_pre_run(self):
        node = live.Atomic.from_recipe(library.my_add.flowrep_recipe)
        d = flowrep_dict.live_to_dict(node, with_io=True)
        self.assertIn("inputs", d)
        self.assertIn("outputs", d)
        self.assertIn("a", d["inputs"])
        self.assertIn("b", d["inputs"])
        # Pre-run: no values populated
        self.assertNotIn("value", d["inputs"]["a"])
        self.assertNotIn("value", d["outputs"]["output"])

    def test_with_io_post_run(self):
        node = wfms.run_recipe(library.my_add.flowrep_recipe, a=3, b=4)
        d = flowrep_dict.live_to_dict(node, with_io=True)
        self.assertEqual(d["inputs"]["a"]["value"], 3)
        self.assertEqual(d["inputs"]["b"]["value"], 4)
        self.assertEqual(d["outputs"]["output"]["value"], 7)

    def test_defaults_included(self):
        node = live.Atomic.from_recipe(library.increment.flowrep_recipe)
        d = flowrep_dict.live_to_dict(node, with_io=True)
        self.assertEqual(d["inputs"]["step"]["default"], 1)
        self.assertNotIn("default", d["inputs"]["x"])

    def test_multi_output(self):
        node = wfms.run_recipe(library.divmod_func.flowrep_recipe, a=17, b=5)
        d = flowrep_dict.live_to_dict(node, with_io=True)
        self.assertIn("quotient", d["outputs"])
        self.assertIn("remainder", d["outputs"])
        self.assertAlmostEqual(d["outputs"]["quotient"]["value"], 3.0)
        self.assertAlmostEqual(d["outputs"]["remainder"]["value"], 2.0)


class TestWorkflowToDict(unittest.TestCase):
    def test_basic_structure(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = live.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node)
        self.assertEqual(d["type"], "workflow")
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        self.assertIn("label", d)
        self.assertNotIn("inputs", d)
        self.assertNotIn("outputs", d)

    def test_label_inferred_from_reference(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = live.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node)
        self.assertEqual(d["label"], "_diamond_workflow")

    def test_label_override(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = live.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node, label="my_label")
        self.assertEqual(d["label"], "my_label")

    def test_child_nodes_present(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = live.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node)
        self.assertIn("my_add_0", d["nodes"])
        self.assertIn("negate_0", d["nodes"])
        self.assertIn("my_mul_0", d["nodes"])
        for child_d in d["nodes"].values():
            self.assertEqual(child_d["type"], "atomic")

    def test_edges_cover_all_recipe_edges(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = live.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node)
        edges = d["edges"]
        # Should have input_edges + sibling edges + output_edges
        n_expected = (
            len(recipe.input_edges) + len(recipe.edges) + len(recipe.output_edges)
        )
        self.assertEqual(len(edges), n_expected)

    def test_edge_format(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = live.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node)
        for src, tgt in d["edges"]:
            self.assertIsInstance(src, str)
            self.assertIsInstance(tgt, str)
            # Every edge string should contain at least one dot
            self.assertIn(".", src)
            self.assertIn(".", tgt)

    def test_with_io_pre_run(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = live.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node, with_io=True)
        self.assertIn("inputs", d)
        self.assertIn("outputs", d)
        self.assertIn("a", d["inputs"])
        self.assertIn("b", d["inputs"])
        # Default from the reference signature
        self.assertEqual(d["inputs"]["b"]["default"], 1)
        self.assertIn("result", d["outputs"])

    def test_with_io_post_run(self):
        wf = wfms.run_recipe(_diamond_workflow.flowrep_recipe, a=3, b=7)
        d = flowrep_dict.live_to_dict(wf, with_io=True)
        self.assertEqual(d["inputs"]["a"]["value"], 3)
        self.assertEqual(d["inputs"]["b"]["value"], 7)
        self.assertEqual(d["outputs"]["result"]["value"], (3 + 7) * (-3))

    def test_child_io_post_run(self):
        """After execution, child nodes also carry values when with_io=True."""
        wf = wfms.run_recipe(_diamond_workflow.flowrep_recipe, a=3, b=7)
        d = flowrep_dict.live_to_dict(wf, with_io=True)
        add_d = d["nodes"]["my_add_0"]
        self.assertEqual(add_d["inputs"]["a"]["value"], 3)
        self.assertEqual(add_d["inputs"]["b"]["value"], 7)
        self.assertEqual(add_d["outputs"]["output"]["value"], 10)

    def test_with_function_on_workflow(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = live.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node, with_function=True)
        # Top-level workflow should have the resolved function
        self.assertIn("function", d)
        self.assertTrue(callable(d["function"]))
        # Children should also have raw callables
        for child_d in d["nodes"].values():
            self.assertTrue(callable(child_d["function"]))

    def test_without_function_uses_metadata(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = live.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node, with_function=False)
        # Top-level: no "function" key (no with_function, reference exists but
        # we only add it when with_function=True)
        self.assertNotIn("function", d)
        # Children: metadata dicts
        for child_d in d["nodes"].values():
            self.assertIsInstance(child_d["function"], dict)

    def test_edges_self_consistent(self):
        """Every port referenced in an edge should correspond to a real node or
        the workflow's own inputs/outputs."""
        recipe = _diamond_workflow.flowrep_recipe
        node = live.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node, with_io=True)

        valid_prefixes = {"inputs", "outputs"} | set(d["nodes"].keys())
        for src, tgt in d["edges"]:
            src_prefix = src.split(".")[0]
            tgt_prefix = tgt.split(".")[0]
            self.assertIn(src_prefix, valid_prefixes, msg=f"bad edge source: {src}")
            self.assertIn(tgt_prefix, valid_prefixes, msg=f"bad edge target: {tgt}")


class TestFlowControlStub(unittest.TestCase):
    def test_raises_not_implemented(self):
        from flowrep.models.nodes import for_model, helper_models
        from flowrep.models import edge_models

        recipe = for_model.ForNode(
            inputs=["xs"],
            outputs=["ys"],
            body_node=helper_models.LabeledNode(
                label="body", node=library.negate.flowrep_recipe
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="body", port="x"
                ): edge_models.InputSource(port="xs")
            },
            output_edges={
                edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
                    node="body", port="output_0"
                )
            },
            nested_ports=["x"],
        )
        fc = live.FlowControl.from_recipe(recipe)
        with self.assertRaises(NotImplementedError):
            flowrep_dict.live_to_dict(fc)


class TestRoundTripConsistency(unittest.TestCase):
    """Verify that pre-run and post-run dicts are structurally compatible."""

    def test_pre_and_post_run_same_keys(self):
        recipe = _diamond_workflow.flowrep_recipe
        pre = flowrep_dict.live_to_dict(live.Workflow.from_recipe(recipe), with_io=True)
        post = flowrep_dict.live_to_dict(
            wfms.run_recipe(recipe, a=3, b=7), with_io=True
        )
        # Same top-level keys
        self.assertEqual(set(pre.keys()), set(post.keys()))
        # Same node labels
        self.assertEqual(set(pre["nodes"].keys()), set(post["nodes"].keys()))
        # Same edges (edges come from the recipe, not execution)
        self.assertEqual(sorted(pre["edges"]), sorted(post["edges"]))

    def test_pre_run_no_values(self):
        recipe = _diamond_workflow.flowrep_recipe
        d = flowrep_dict.live_to_dict(live.Workflow.from_recipe(recipe), with_io=True)
        for port_d in d["outputs"].values():
            self.assertNotIn("value", port_d)

    def test_post_run_has_values(self):
        d = flowrep_dict.live_to_dict(
            wfms.run_recipe(_diamond_workflow.flowrep_recipe, a=3, b=7),
            with_io=True,
        )
        for port_d in d["outputs"].values():
            self.assertIn("value", port_d)


if __name__ == "__main__":
    unittest.main()
