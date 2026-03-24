"""Tests for the live → nested-dict converter."""

import dataclasses
import math
import unittest

import networkx as nx
from flowrep.models.api import live, parsers, schemas, wfms

from semantikon import flowrep_dict


@parsers.atomic
def my_add(a, b):
    return a + b


def my_mul(a, b):
    return a * b


@parsers.atomic
def negate(x):
    return -x


# Reuse the diamond recipe from the test suite — it has a reference, so we get
# annotations and defaults on the live Workflow.
@parsers.workflow
def _diamond_workflow(a: int, b: int = 1) -> int:
    s = my_add(a, b)
    n = negate(a)
    result = my_mul(s, n)
    return result


@parsers.workflow
def _passthrough_workflow(x):
    y = negate(x)
    return x, y


@parsers.atomic
def increment(x, step=1):
    return x + step


@parsers.atomic
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


@parsers.workflow
def workflow_with_data(a=10, b=20):
    x = add(a, b)
    y = multiply(x, b)
    return x, y


@parsers.workflow
def example_macro(a=10, b=20):
    c, d = operation(a, b)
    e = add(c, y=d)
    f = multiply(e)
    return f


@parsers.workflow
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


@parsers.workflow
def workflow_with_class(test: TestClass):
    test = some_function(test)
    return test


def example_function(x):
    """An example function to be hashed."""
    return x * 2


@parsers.atomic
def identity(x):
    return x


def _inner_workflow_with_output_0() -> schemas.WorkflowNode:
    """A workflow whose sole output is named ``output_0``.

    This is unusual (the workflow parser names outputs from return variables),
    but perfectly valid for manually constructed recipes.
    """
    return schemas.WorkflowNode(
        inputs=["x"],
        outputs=["output_0"],
        nodes={"identity_0": identity.flowrep_recipe},
        input_edges={
            schemas.TargetHandle(node="identity_0", port="x"): schemas.InputSource(
                port="x"
            ),
        },
        edges={},
        output_edges={
            schemas.OutputTarget(port="output_0"): schemas.SourceHandle(
                node="identity_0", port="x"
            ),
        },
    )


def _parent_workflow(inner: schemas.WorkflowNode) -> schemas.WorkflowNode:
    """A workflow that wraps *inner* and reads its ``output_0`` port."""
    return schemas.WorkflowNode(
        inputs=["x"],
        outputs=["result"],
        nodes={"inner_0": inner},
        input_edges={
            schemas.TargetHandle(node="inner_0", port="x"): schemas.InputSource(
                port="x"
            ),
        },
        edges={},
        output_edges={
            schemas.OutputTarget(port="result"): schemas.SourceHandle(
                node="inner_0", port="output_0"
            ),
        },
    )


class TestNonNodeToDict(unittest.TestCase):
    def test_non_node_type_raises(self):
        with self.assertRaisesRegex(TypeError, "Unsupported live node type"):
            flowrep_dict.live_to_dict(123)


class TestAtomicToDict(unittest.TestCase):
    def test_basic_structure(self):
        node = schemas.Atomic.from_recipe(my_add.flowrep_recipe)
        d = flowrep_dict.live_to_dict(node)
        self.assertEqual(d["type"], "atomic")
        self.assertIn("function", d)
        # Default: metadata dict, not raw callable
        self.assertIsInstance(d["function"], dict)
        self.assertIn("module", d["function"])
        self.assertNotIn("inputs", d)
        self.assertNotIn("outputs", d)

    def test_with_function(self):
        node = schemas.Atomic.from_recipe(my_add.flowrep_recipe)
        d = flowrep_dict.live_to_dict(node, with_function=True)
        self.assertTrue(callable(d["function"]))

    def test_with_io_pre_run(self):
        node = schemas.Atomic.from_recipe(my_add.flowrep_recipe)
        d = flowrep_dict.live_to_dict(node, with_io=True)
        self.assertIn("inputs", d)
        self.assertIn("outputs", d)
        self.assertIn("a", d["inputs"])
        self.assertIn("b", d["inputs"])
        # Pre-run: no values populated
        self.assertNotIn("value", d["inputs"]["a"])
        self.assertNotIn("value", d["outputs"]["output"])

    def test_with_io_post_run(self):
        node = wfms.run_recipe(my_add.flowrep_recipe, a=3, b=4)
        d = flowrep_dict.live_to_dict(node, with_io=True)
        self.assertEqual(d["inputs"]["a"]["value"], 3)
        self.assertEqual(d["inputs"]["b"]["value"], 4)
        self.assertEqual(d["outputs"]["output"]["value"], 7)

    def test_defaults_included(self):
        node = schemas.Atomic.from_recipe(increment.flowrep_recipe)
        d = flowrep_dict.live_to_dict(node, with_io=True)
        self.assertEqual(d["inputs"]["step"]["default"], 1)
        self.assertNotIn("default", d["inputs"]["x"])

    def test_multi_output(self):
        node = wfms.run_recipe(divmod_func.flowrep_recipe, a=17, b=5)
        d = flowrep_dict.live_to_dict(node, with_io=True)
        self.assertIn("quotient", d["outputs"])
        self.assertIn("remainder", d["outputs"])
        self.assertAlmostEqual(d["outputs"]["quotient"]["value"], 3.0)
        self.assertAlmostEqual(d["outputs"]["remainder"]["value"], 2.0)


class TestWorkflowToDict(unittest.TestCase):
    def test_basic_structure(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = schemas.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node)
        self.assertEqual(d["type"], "workflow")
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        self.assertIn("label", d)
        self.assertNotIn("inputs", d)
        self.assertNotIn("outputs", d)

    def test_label_inferred_from_reference(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = schemas.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node)
        self.assertEqual(d["label"], "_diamond_workflow")

    def test_label_override(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = schemas.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node, label="my_label")
        self.assertEqual(d["label"], "my_label")

    def test_child_nodes_present(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = schemas.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node)
        self.assertIn("my_add_0", d["nodes"])
        self.assertIn("negate_0", d["nodes"])
        self.assertIn("my_mul_0", d["nodes"])
        for child_d in d["nodes"].values():
            self.assertEqual(child_d["type"], "atomic")

    def test_edges_cover_all_recipe_edges(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = schemas.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node)
        edges = d["edges"]
        # Should have input_edges + sibling edges + output_edges
        n_expected = (
            len(recipe.input_edges) + len(recipe.edges) + len(recipe.output_edges)
        )
        self.assertEqual(len(edges), n_expected)

    def test_passthrough_edges(self):
        recipe = _passthrough_workflow.flowrep_recipe
        node = schemas.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node)
        self.assertIn(("inputs.x", "outputs.x"), d["edges"])

    def test_edge_format(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = schemas.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node)
        for src, tgt in d["edges"]:
            self.assertIsInstance(src, str)
            self.assertIsInstance(tgt, str)
            # Every edge string should contain at least one dot
            self.assertIn(".", src)
            self.assertIn(".", tgt)

    def test_with_io_pre_run(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = schemas.Workflow.from_recipe(recipe)
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
        node = schemas.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node, with_function=True)
        # Top-level workflow should have the resolved function
        self.assertIn("function", d)
        self.assertTrue(callable(d["function"]))
        # Children should also have raw callables
        for child_d in d["nodes"].values():
            self.assertTrue(callable(child_d["function"]))

    def test_without_function_uses_metadata(self):
        recipe = _diamond_workflow.flowrep_recipe
        node = schemas.Workflow.from_recipe(recipe)
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
        node = schemas.Workflow.from_recipe(recipe)
        d = flowrep_dict.live_to_dict(node, with_io=True)

        valid_prefixes = {"inputs", "outputs"} | set(d["nodes"].keys())
        for src, tgt in d["edges"]:
            src_prefix = src.split(".")[0]
            tgt_prefix = tgt.split(".")[0]
            self.assertIn(src_prefix, valid_prefixes, msg=f"bad edge source: {src}")
            self.assertIn(tgt_prefix, valid_prefixes, msg=f"bad edge target: {tgt}")


class TestFlowControlStub(unittest.TestCase):
    def test_raises_not_implemented(self):
        recipe = schemas.ForNode(
            inputs=["xs"],
            outputs=["ys"],
            body_node=schemas.LabeledNode(label="body", node=negate.flowrep_recipe),
            input_edges={
                schemas.TargetHandle(node="body", port="x"): schemas.InputSource(
                    port="xs"
                )
            },
            output_edges={
                schemas.OutputTarget(port="ys"): schemas.SourceHandle(
                    node="body", port="output_0"
                )
            },
            nested_ports=["x"],
        )
        fc = schemas.FlowControl.from_recipe(recipe)
        with self.assertRaises(NotImplementedError):
            flowrep_dict.live_to_dict(fc)


class TestRoundTripConsistency(unittest.TestCase):
    """Verify that pre-run and post-run dicts are structurally compatible."""

    def test_pre_and_post_run_same_keys(self):
        recipe = _diamond_workflow.flowrep_recipe
        pre = flowrep_dict.live_to_dict(
            schemas.Workflow.from_recipe(recipe), with_io=True
        )
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
        d = flowrep_dict.live_to_dict(
            schemas.Workflow.from_recipe(recipe), with_io=True
        )
        for port_d in d["outputs"].values():
            self.assertNotIn("value", port_d)

    def test_post_run_has_values(self):
        d = flowrep_dict.live_to_dict(
            wfms.run_recipe(_diamond_workflow.flowrep_recipe, a=3, b=7),
            with_io=True,
        )
        for port_d in d["outputs"].values():
            self.assertIn("value", port_d)


class TestDigraphConverters(unittest.TestCase):
    def test_wf_dict_to_graph(self):
        # wf_dict = example_workflow.get_flowrep_dict()
        wf_dict = flowrep_dict.live_to_dict(
            live.recipe2live(example_workflow.flowrep_recipe),
            with_io=False,
            with_function=True,
        )
        G = flowrep_dict._get_workflow_graph(wf_dict)
        self.assertIsInstance(G, nx.DiGraph)
        with self.assertRaises(ValueError):
            G = flowrep_dict._get_workflow_graph(wf_dict)
            _ = flowrep_dict._simple_run(G)

        wf_dict = flowrep_dict.live_to_dict(
            live.recipe2live(example_workflow.flowrep_recipe),
            with_io=True,
            with_function=True,
        )
        wf_dict["inputs"] = {"a": {"value": 1}, "b": {"default": 2}}
        wf_dict["nodes"]["add_0"]["inputs"] = {"y": {"metadata": "something"}}
        G = flowrep_dict._get_workflow_graph(wf_dict)
        self.assertDictEqual(
            G.nodes["add_0:inputs@y"],
            {"metadata": "something", "position": 0, "step": "input"},
        )
        G = flowrep_dict._simple_run(G)
        self.assertDictEqual(G.nodes["outputs@z"], {"step": "output", "value": 12})
        rev_edges = flowrep_dict._graph_to_wf_dict(G)["edges"]
        self.assertEqual(
            sorted(rev_edges),
            sorted(wf_dict["edges"]),
        )
        rev_macro_edges = flowrep_dict._graph_to_wf_dict(G)["nodes"]["example_macro_0"][
            "edges"
        ]
        self.assertEqual(
            sorted(rev_macro_edges),
            sorted(wf_dict["nodes"]["example_macro_0"]["edges"]),
        )

    def test_get_hashed_node_dict(self):

        # workflow_dict = workflow_with_data.run(a=10, b=20)
        workflow_dict = flowrep_dict.live_to_dict(
            wfms.run_recipe(workflow_with_data.flowrep_recipe, a=10, b=20),
            with_io=True,
            with_function=True,
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
        workflow_dict = flowrep_dict.live_to_dict(
            live.recipe2live(workflow_with_data.flowrep_recipe),
            with_io=True,
            with_function=True,
        )
        hashed_dict = flowrep_dict.get_hashed_node_dict(workflow_dict)
        for node in hashed_dict.values():
            self.assertNotIn("hash", node)
        workflow_dict["inputs"] = {"a": {"value": 10}, "b": {"value": 20}}
        # workflow_dict_run = workflow_with_data.run(a=10, b=20)
        workflow_dict_run = flowrep_dict.live_to_dict(
            wfms.run_recipe(workflow_with_data.flowrep_recipe, a=10, b=20),
            with_io=True,
            with_function=True,
        )
        self.assertDictEqual(
            flowrep_dict.get_hashed_node_dict(workflow_dict),
            flowrep_dict.get_hashed_node_dict(workflow_dict_run),
        )

        # workflow_dict = example_workflow.run(a=10, b=20)
        workflow_dict = flowrep_dict.live_to_dict(
            wfms.run_recipe(example_workflow.flowrep_recipe, a=10, b=20),
            with_io=True,
            with_function=True,
        )
        hashed_dict = flowrep_dict.get_hashed_node_dict(workflow_dict)
        self.assertIn("example_macro_0.operation_0", hashed_dict)

        test_instance = TestClass()
        # workflow_dict = workflow_with_class.run(test=test_instance)
        workflow_dict = flowrep_dict.live_to_dict(
            wfms.run_recipe(workflow_with_class.flowrep_recipe, test=test_instance),
            with_io=True,
            with_function=True,
        )
        hashed_dict = flowrep_dict.get_hashed_node_dict(workflow_dict)
        for node in hashed_dict.values():
            self.assertIn("hash", node)
            self.assertIsInstance(node["hash"], str)
            self.assertEqual(len(node["hash"]), 64)


class TestTools(unittest.TestCase):
    def test_defaultdict(self):
        d = flowrep_dict.recursive_defaultdict()
        d["x"]["y"]["z"] = 3
        self.assertEqual(d["x"]["y"]["z"], 3)
        normal = {"a": {"b": {"c": 1}}}
        dd = flowrep_dict.dict_to_recursive_dd(normal)
        self.assertEqual(dd["a"]["b"]["c"], 1)
        dd["x"]["y"]["z"] = 2
        self.assertEqual(dd["x"]["y"]["z"], 2)
        plain = flowrep_dict.recursive_dd_to_dict(dd)
        self.assertEqual(plain["a"]["b"]["c"], 1)

    def test_get_function_metadata(self):
        meta = flowrep_dict.get_function_metadata(example_function, full_metadata=True)
        self.assertIn("name", meta)
        self.assertIn("module", meta)
        self.assertIn("docstring", meta)
        self.assertEqual(meta["name"], "example_function")
        meta = flowrep_dict.get_function_metadata(example_function, full_metadata=False)
        self.assertNotIn("docstring", meta)

    def test_hash_function(self):
        import math

        # Test built-in function (triggers except path - no source code available)
        self.assertEqual(
            flowrep_dict.hash_function(math.sin)[:40],
            "sin:0146c21ab456a735f07d62b456f003ce3dc6",
        )

        # Test regular function with source code available
        expected_hash = "example_function:196938631e98c05e128b0b1"
        self.assertEqual(
            flowrep_dict.hash_function(example_function)[: len(expected_hash)],
            expected_hash,
        )

    def test_hash_function_fallback(self):
        """Test that hash_function falls back to signature for functions without source."""

        # Built-in functions (not Python functions, so uses else branch)
        hash_result = flowrep_dict.hash_function(math.sin)
        self.assertTrue(hash_result.startswith("sin:"))
        self.assertEqual(len(hash_result), 68)  # "sin:" + 64 char hex hash

        # Test another built-in
        hash_result = flowrep_dict.hash_function(len)
        self.assertTrue(hash_result.startswith("len:"))
        self.assertEqual(len(hash_result), 68)

    def test_hash_function_except_path(self):
        """Test that hash_function handles functions where source is unavailable (except path)."""
        # Create a lambda function - it's a function but source is unavailable
        hash_result = flowrep_dict.hash_function(lambda x: x * 2)
        self.assertTrue(hash_result.startswith("<lambda>:"))
        self.assertEqual(len(hash_result), 73)  # "<lambda>:" + 64 char hex hash

        # Create a dynamically defined function using exec
        exec_globals = {}
        exec("def dynamic_test_func(x):\n    return x + 1", exec_globals)
        dynamic_func = exec_globals["dynamic_test_func"]
        hash_result = flowrep_dict.hash_function(dynamic_func)
        self.assertTrue(hash_result.startswith("dynamic_test_func:"))
        self.assertEqual(
            len(hash_result), 82
        )  # "dynamic_test_func:" + 64 char hex hash


class TestOutputSanitizationConsistency(unittest.TestCase):
    """Edges and output dicts must agree on port names after sanitization."""

    def test_wfms_runs_fine(self):
        """The recipe itself is well-formed and executes correctly."""
        inner = _inner_workflow_with_output_0()
        parent = _parent_workflow(inner)
        node = wfms.run_recipe(parent, x=42)
        self.assertEqual(node.output_ports["result"].value, 42)

    def test_inner_dict_edges_consistent_with_outputs(self):
        """The inner workflow's edges and outputs dict must use the same name."""
        inner = _inner_workflow_with_output_0()
        inner_live = schemas.Workflow.from_recipe(inner)
        d = flowrep_dict.live_to_dict(inner_live, with_io=True, with_function=True)

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
        parent_live = schemas.Workflow.from_recipe(parent)
        d = flowrep_dict.live_to_dict(parent_live, with_io=True, with_function=True)
        G = flowrep_dict._get_workflow_graph(d)

        inner_output_nodes = [n for n in G.nodes if n.startswith("inner_0:outputs@")]
        output_names = [n.split("@")[-1] for n in inner_output_nodes]
        self.assertEqual(
            output_names,
            ["output"],
            msg="Should be exactly one inner output node named 'output'",
        )

    def test_nested_dict_runs_via_simple_run(self):
        """The full dict → nx graph → simple_run path must produce the right value."""
        inner = _inner_workflow_with_output_0()
        parent = _parent_workflow(inner)
        parent_live = schemas.Workflow.from_recipe(parent)
        d = flowrep_dict.live_to_dict(parent_live, with_io=True, with_function=True)
        d["inputs"]["x"]["value"] = 42

        G = flowrep_dict._get_workflow_graph(d)
        G = flowrep_dict._simple_run(G)
        self.assertEqual(G.nodes["outputs@result"]["value"], 42)


if __name__ == "__main__":
    unittest.main()
