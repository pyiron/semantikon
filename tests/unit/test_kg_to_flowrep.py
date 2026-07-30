import unittest
from unittest.mock import MagicMock

import flowrep as fr
import networkx as nx
from rdflib import RDFS, Graph, Literal, URIRef

from semantikon import get_knowledge_graph, kg2recipe
from semantikon import kg_to_flowrep as kgf
from semantikon.workflow import workflow


def add_one(a):
    return a + 1


def add(x, y):
    return x + y


@workflow
def add_more(x):
    y = add_one(x)
    z = add(x, y)
    return z


@workflow
def my_workflow(x=1, y=2):
    z = add_one(x)
    q = add_more(z)
    result = add(q, y)
    return result


def times_two(x):
    return x * 2


@workflow
def multiply_by_two(x=2):
    result = times_two(x)
    return result


@fr.workflow
def double_via_constant(x):
    doubled = fr.std.mul(2, x)
    return doubled


def pick(options: list, index: int) -> int:
    return options[index]


@fr.workflow
def pick_from_constant_list(i):
    chosen = pick([10, 20, 30], i)
    return chosen


@fr.workflow
def compare_to_constant_none(x):
    missing = fr.std.is_(x, None)
    return missing


class TestKgToFlowrep(unittest.TestCase):
    def test_round_trip_with_a_constant_node(self):
        graph = get_knowledge_graph(double_via_constant.flowrep_recipe)
        reconstructed = kg2recipe(graph)

        constants = {
            label: node
            for label, node in reconstructed.nodes.items()
            if isinstance(node, fr.schemas.ConstantRecipe)
        }
        self.assertEqual(
            list(constants),
            ["constant_0"],
            msg="The constant node should survive the round trip as a constant, "
            "not be dropped or demoted to an atomic node",
        )
        self.assertEqual(
            constants["constant_0"].constant,
            2,
            msg="The reconstructed constant should carry the original value",
        )

        original = fr.tools.run_recipe(double_via_constant.flowrep_recipe, x=3)
        converted = fr.tools.run_recipe(reconstructed, x=3)
        self.assertEqual(
            original.output_ports["doubled"].value,
            converted.output_ports["doubled"].value,
            msg="The reconstructed recipe should compute what the original does",
        )

    def test_round_trip_from_knowledge_graph(self):
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        reconstructed = kg2recipe(graph)

        original_result = fr.tools.run_recipe(my_workflow.flowrep_recipe, x=3, y=5)
        converted_result = fr.tools.run_recipe(reconstructed, x=3, y=5)
        self.assertEqual(
            original_result.output_ports["result"].value,
            converted_result.output_ports["result"].value,
        )

    def test_round_trip_with_a_container_constant(self):
        graph = get_knowledge_graph(pick_from_constant_list.flowrep_recipe)
        # Serialize and reparse: an in-memory rdflib Literal secretly keeps the
        # original Python object, so only a persisted graph proves the value
        # actually survives as RDF.
        graph = Graph().parse(data=graph.serialize(format="turtle"), format="turtle")
        reconstructed = kg2recipe(graph)

        (constant,) = [
            node
            for node in reconstructed.nodes.values()
            if isinstance(node, fr.schemas.ConstantRecipe)
        ]
        self.assertEqual(
            constant.constant,
            [10, 20, 30],
            msg="A JSONable container constant should come back as the container "
            "itself, not as its string repr",
        )

        original = fr.tools.run_recipe(pick_from_constant_list.flowrep_recipe, i=1)
        converted = fr.tools.run_recipe(reconstructed, i=1)
        self.assertEqual(
            original.output_ports["chosen"].value,
            converted.output_ports["chosen"].value,
        )

    def test_round_trip_with_a_none_constant(self):
        graph = get_knowledge_graph(compare_to_constant_none.flowrep_recipe)
        graph = Graph().parse(data=graph.serialize(format="turtle"), format="turtle")
        reconstructed = kg2recipe(graph)

        (constant,) = [
            node
            for node in reconstructed.nodes.values()
            if isinstance(node, fr.schemas.ConstantRecipe)
        ]
        self.assertIsNone(
            constant.constant,
            msg="A None constant should come back as None, not the string 'None'",
        )

    def test_requires_disambiguation_for_multiple_workflows(self):
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        graph += get_knowledge_graph(multiply_by_two.flowrep_recipe, prefix="T")

        with self.assertRaisesRegex(ValueError, "multiple root workflows"):
            _ = kg2recipe(graph)

        reconstructed = kg2recipe(graph, workflow_name="multiply_by_two")
        converted_result = fr.tools.run_recipe(reconstructed, x=7)
        self.assertEqual(converted_result.output_ports["result"].value, 14)

    def test_requires_t_box_information(self):
        graph = get_knowledge_graph(my_workflow.flowrep_recipe, include_t_box=False)
        with self.assertRaisesRegex(ValueError, "No workflow nodes found in graph"):
            _ = kg2recipe(graph)

    def test_identifier_and_label_fallbacks(self):
        graph = Graph()
        node = URIRef("http://example.org/node")

        self.assertEqual(kgf._identifier(graph, node), str(node))
        self.assertEqual(kgf._label(graph, node), str(node))

        graph.add((node, RDFS.label, Literal("my_label")))
        self.assertEqual(kgf._identifier(graph, node), "my_label")
        self.assertEqual(kgf._label(graph, node), "my_label")

        graph.add((node, kgf.SNS.local_identifier, Literal("my_identifier")))
        self.assertEqual(kgf._identifier(graph, node), "my_identifier")
        self.assertEqual(kgf._label(graph, node), "my_label")

    def test_select_workflow_variants(self):
        graph = Graph()
        uri_1 = URIRef("http://example.org/wf1")
        uri_2 = URIRef("http://example.org/wf2")
        graph.add((uri_1, kgf.SNS.local_identifier, Literal("wf_a")))
        graph.add((uri_2, kgf.SNS.local_identifier, Literal("wf_b")))

        roots = {uri_1: "label_a", uri_2: "label_b"}
        workflows = roots.values()

        self.assertEqual(
            kgf._select_workflow(graph, roots, workflows, workflow_name="label_a"),
            "label_a",
        )
        self.assertEqual(
            kgf._select_workflow(graph, roots, workflows, workflow_name="wf_a"),
            "label_a",
        )
        with self.assertRaisesRegex(ValueError, "Unknown workflow"):
            _ = kgf._select_workflow(graph, roots, workflows, workflow_name="missing")
        with self.assertRaisesRegex(ValueError, "multiple root workflows"):
            _ = kgf._select_workflow(graph, roots, workflows, workflow_name=None)

        ambiguous_graph = Graph()
        ambiguous_uri_1 = URIRef("http://example.org/wf_same_1")
        ambiguous_uri_2 = URIRef("http://example.org/wf_same_2")
        for uri in (ambiguous_uri_1, ambiguous_uri_2):
            ambiguous_graph.add((uri, kgf.SNS.local_identifier, Literal("wf_same")))
        ambiguous_roots = {ambiguous_uri_1: "label_1", ambiguous_uri_2: "label_2"}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _ = kgf._select_workflow(
                ambiguous_graph,
                ambiguous_roots,
                ambiguous_roots.values(),
                workflow_name="wf_same",
            )

    def test_split_by_roots_errors(self):
        graph = Graph()

        # Test 1: Graph with no roots
        no_root_graph = nx.DiGraph()
        no_root_graph.add_node(URIRef("http://example.org/unrelated"))
        with self.assertRaisesRegex(ValueError, "Could not assign"):
            _ = kgf._split_by_roots(graph, no_root_graph, roots={})

        # Test 2: Multiple roots in same component
        multi_root_graph = nx.DiGraph()
        root_a = URIRef("http://example.org/root_a")
        root_b = URIRef("http://example.org/root_b")
        multi_root_graph.add_edge(root_a, root_b)
        with self.assertRaisesRegex(ValueError, "more than one root workflow"):
            _ = kgf._split_by_roots(
                graph,
                multi_root_graph,
                roots={
                    root_a: "root_a",
                    root_b: "root_b",
                },
            )

    def test_add_io_nodes_error_paths(self):
        node = URIRef("http://example.org/node")
        io_node = URIRef("http://example.org/io")
        function_node = URIRef("http://example.org/function")
        function_dict = {
            function_node: {
                "data": {"qualname": "f"},
                "input_args": [{"arg": "x"}],
                "output_args": [{"arg": "x"}],
            }
        }
        node_function_dict = {node: function_node}

        graph = MagicMock()
        graph.query.return_value = [(node, io_node)]
        graph.objects.return_value = []
        workflow_graph = nx.DiGraph()
        with self.assertRaisesRegex(ValueError, "Expected one local identifier"):
            kgf._add_io_nodes(
                graph,
                workflow_graph,
                function_dict,
                node_function_dict,
                io_type="input",
            )

        graph.objects.return_value = [Literal("unknown_arg")]
        with self.assertRaisesRegex(ValueError, "Could not match argument"):
            kgf._add_io_nodes(
                graph,
                workflow_graph,
                function_dict,
                node_function_dict,
                io_type="input",
            )

    def test_reorganize_and_reconnect_helpers(self):
        graph = nx.DiGraph()
        p1 = URIRef("http://example.org/p1")
        p2 = URIRef("http://example.org/p2")
        n1 = URIRef("http://example.org/n1")
        n2 = URIRef("http://example.org/n2")
        data = URIRef("http://example.org/data")
        graph.add_edges_from([(p1, n1), (p2, n2), (n1, data), (n2, data)])
        kgf._reorganize_output_edges(graph, data, position={p1: 0, p2: 1})
        self.assertEqual(len(list(graph.predecessors(data))), 1)

        s1 = URIRef("http://example.org/s1")
        s2 = URIRef("http://example.org/s2")
        i1 = URIRef("http://example.org/i1")
        i2 = URIRef("http://example.org/i2")
        in_data = URIRef("http://example.org/in_data")
        graph = nx.DiGraph()
        graph.add_edges_from(
            [(in_data, i1), (in_data, i2), (i1, s1), (i2, s2), (s1, s2)]
        )
        kgf._reorganize_input_edges(graph, in_data, position={s1: 0, s2: 1})
        self.assertEqual(len(list(graph.successors(in_data))), 1)

        reconnect = nx.DiGraph()
        out_node = URIRef("http://example.org/out")
        input_one = URIRef("http://example.org/in1")
        input_two = URIRef("http://example.org/in2")
        data_node = URIRef("http://example.org/d")
        reconnect.add_edges_from(
            [(out_node, data_node), (data_node, input_one), (data_node, input_two)]
        )
        kgf._reconnect_io(reconnect, data_node)
        self.assertIn((out_node, input_one), reconnect.edges)
        self.assertIn((out_node, input_two), reconnect.edges)

    def test_kg2recipe_with_uriref_workflow_name(self):
        """Test that kg2recipe accepts URIRef workflow_name parameter."""
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        roots = kgf._workflow_roots(graph)

        # Get the URIRef for the workflow
        workflow_uriref = next(iter(roots.keys()))

        # Should work with URIRef as workflow_name
        reconstructed = kg2recipe(graph, workflow_name=workflow_uriref)

        # Verify round-trip correctness by running the recipe
        original_result = fr.tools.run_recipe(my_workflow.flowrep_recipe, x=3, y=5)
        converted_result = fr.tools.run_recipe(reconstructed, x=3, y=5)
        self.assertEqual(
            original_result.output_ports["result"].value,
            converted_result.output_ports["result"].value,
        )

    def test_select_workflow_with_invalid_uriref(self):
        """Test that _select_workflow raises error for unknown URIRef."""
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        roots = kgf._workflow_roots(graph)
        workflow_graph = kgf._build_workflow_graph(graph)
        workflows = kgf._split_by_roots(graph, workflow_graph, roots)

        # Create a non-existent URIRef
        invalid_uriref = URIRef("http://example.org/nonexistent")

        with self.assertRaises(ValueError) as context:
            kgf._select_workflow(
                graph, roots, workflows.keys(), workflow_name=invalid_uriref
            )

        self.assertIn("Unknown workflow URIRef", str(context.exception))

    def test_select_workflow_with_string_vs_uriref(self):
        """Test that string and URIRef selection produce same result."""
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        roots = kgf._workflow_roots(graph)
        workflow_graph = kgf._build_workflow_graph(graph)
        workflows = kgf._split_by_roots(graph, workflow_graph, roots)
        # Get the workflow label and URIRef (now keys are URIRefs, values are labels)
        workflow_uriref = next(iter(roots.keys()))
        workflow_label = roots[workflow_uriref]

        # Both should select the same workflow
        result_from_string = kgf._select_workflow(
            graph, roots, workflows.keys(), workflow_name=workflow_label
        )
        result_from_uriref = kgf._select_workflow(
            graph, roots, workflows.keys(), workflow_name=workflow_uriref
        )

        self.assertEqual(result_from_string, result_from_uriref)


if __name__ == "__main__":
    unittest.main()
