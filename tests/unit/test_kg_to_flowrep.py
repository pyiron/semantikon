import unittest

import flowrep as fr
from rdflib import RDFS, Graph, Literal, URIRef

from semantikon import get_knowledge_graph, kg2recipe
from semantikon import kg_to_flowrep as kgf
from semantikon import ontology as onto


def add_one(a):
    return a + 1


def add(x, y):
    return x + y


@fr.workflow
def add_more(x):
    y = add_one(x)
    z = add(x, y)
    return z


@fr.workflow
def my_workflow(x=1, y=2):
    z = add_one(x)
    q = add_more(z)
    result = add(q, y)
    return result


def times_two(x):
    return x * 2


@fr.workflow
def multiply_by_two(x=2):
    result = times_two(x)
    return result


@fr.workflow
def double_via_constant(x):
    doubled = fr.std.mul(2, x)
    return doubled


@fr.workflow
def multiply_then_add(x, y):
    a = fr.std.mul(3, x)
    b = fr.std.mul(5, y)
    result = fr.std.add(a, b)
    return result


@fr.workflow
def weighted_sum(x, y, z):
    a = fr.std.mul(10, x)
    b = fr.std.mul(20, y)
    c = fr.std.mul(30, z)
    sum_ab = fr.std.add(a, b)
    result = fr.std.add(sum_ab, c)
    return result


class TestKgToFlowrep(unittest.TestCase):
    def test_round_trip_from_knowledge_graph(self):
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        reconstructed = kg2recipe(graph)

        original_result = fr.tools.run_recipe(my_workflow.flowrep_recipe, x=3, y=5)
        converted_result = fr.tools.run_recipe(reconstructed, x=3, y=5)
        self.assertEqual(
            original_result.output_ports["result"].value,
            converted_result.output_ports["result"].value,
        )

    def test_requires_disambiguation_for_multiple_workflows(self):
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        graph += get_knowledge_graph(multiply_by_two.flowrep_recipe, prefix="T")

        with self.assertRaisesRegex(ValueError, "multiple root workflows"):
            _ = kg2recipe(graph)

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

    def test_add_io_nodes_error_paths(self):
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        with self.assertRaisesRegex(ValueError, "Unknown workflow"):
            kg2recipe(graph, workflow_name=URIRef("http://example.org/node"))

    def test_kg2recipe_with_uriref_workflow_name(self):
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        reconstructed = kg2recipe(graph, workflow_name="my_workflow")
        converted_result = fr.tools.run_recipe(reconstructed, x=3, y=5)
        self.assertEqual(converted_result.output_ports["result"].value, 14)

    def test_select_workflow_with_invalid_uriref(self):
        """Test that _select_workflow raises error for unknown URIRef."""
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        with self.assertRaisesRegex(ValueError, "Unknown workflow"):
            kg2recipe(graph, workflow_name=URIRef("http://example.org/nonexistent"))

    def test_select_workflow_with_string_vs_uriref(self):
        """Test that string and URIRef selection produce same result."""
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        reconstructed_from_string = kg2recipe(graph, workflow_name="my_workflow")
        reconstructed_from_uri = kg2recipe(
            graph,
            workflow_name=URIRef("http://pyiron.org/ontology/Wcfff5bb1_my_workflow"),
        )
        self.assertEqual(
            fr.tools.run_recipe(reconstructed_from_string, x=3, y=5)
            .output_ports["result"]
            .value,
            fr.tools.run_recipe(reconstructed_from_uri, x=3, y=5)
            .output_ports["result"]
            .value,
        )

    def test_kg_to_recipe_with_constants(self):
        """Test converting knowledge graph back to recipe with constants preserved."""
        graph = get_knowledge_graph(double_via_constant.flowrep_recipe)
        reconstructed = kg2recipe(graph)

        # Verify the reconstructed recipe has the constant node
        self.assertIn("constant_0", reconstructed.nodes)
        self.assertIsInstance(
            reconstructed.nodes["constant_0"], fr.schemas.ConstantRecipe
        )
        self.assertEqual(reconstructed.nodes["constant_0"].constant, 2)

    def test_kg_to_recipe_constant_execution(self):
        """Test that reconstructed recipe with constants executes correctly."""
        graph = get_knowledge_graph(double_via_constant.flowrep_recipe)
        reconstructed = kg2recipe(graph)

        # Execute both original and reconstructed
        original_result = fr.tools.run_recipe(double_via_constant.flowrep_recipe, x=5)
        reconstructed_result = fr.tools.run_recipe(reconstructed, x=5)

        # Compare results
        original_value = original_result.output_ports["doubled"].value
        reconstructed_value = reconstructed_result.output_ports["doubled"].value
        self.assertEqual(original_value, reconstructed_value)
        self.assertEqual(original_value, 10)

    def test_kg_to_recipe_multiple_constants(self):
        """Test knowledge graph round-trip with multiple constants."""
        graph = get_knowledge_graph(multiply_then_add.flowrep_recipe)
        reconstructed = kg2recipe(graph)

        # Verify the reconstructed recipe has the constant nodes
        self.assertIn("constant_0", reconstructed.nodes)
        self.assertIn("constant_1", reconstructed.nodes)
        self.assertIsInstance(
            reconstructed.nodes["constant_0"], fr.schemas.ConstantRecipe
        )
        self.assertIsInstance(
            reconstructed.nodes["constant_1"], fr.schemas.ConstantRecipe
        )

        const_values = {
            reconstructed.nodes["constant_0"].constant,
            reconstructed.nodes["constant_1"].constant,
        }
        self.assertEqual(const_values, {3, 5})

    def test_kg_to_recipe_multiple_constants_execution(self):
        """Test execution of reconstructed recipe with multiple constants."""
        graph = get_knowledge_graph(multiply_then_add.flowrep_recipe)
        reconstructed = kg2recipe(graph)

        # Execute with test values
        test_x, test_y = 2, 4
        expected = 3 * test_x + 5 * test_y  # 26

        original_result = fr.tools.run_recipe(
            multiply_then_add.flowrep_recipe, x=test_x, y=test_y
        )
        reconstructed_result = fr.tools.run_recipe(reconstructed, x=test_x, y=test_y)

        original_value = original_result.output_ports["result"].value
        reconstructed_value = reconstructed_result.output_ports["result"].value

        self.assertEqual(original_value, expected)
        self.assertEqual(reconstructed_value, expected)
        self.assertEqual(original_value, reconstructed_value)

    def test_kg_round_trip_preserves_constants(self):
        """Test that constants are preserved through KG->Recipe->KG round-trip."""
        graph1 = get_knowledge_graph(double_via_constant.flowrep_recipe)
        reconstructed = kg2recipe(graph1)
        graph2 = get_knowledge_graph(reconstructed)

        # Query both graphs for constants
        const_query = """\
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX pmdco: <https://w3id.org/pmd/co/PMD_>

        SELECT ?value
        WHERE {
            ?subject a ?class .
            ?class rdfs:subClassOf ?restriction .
            ?restriction a owl:Restriction .
            ?restriction owl:onProperty pmdco:0000006 .
            ?restriction owl:hasValue ?value .
        }
        """

        constants_in_graph1 = sorted(
            [str(row.value) for row in graph1.query(const_query)]
        )
        constants_in_graph2 = sorted(
            [str(row.value) for row in graph2.query(const_query)]
        )

        # Both should have the same constants
        self.assertEqual(constants_in_graph1, constants_in_graph2)
        self.assertEqual(constants_in_graph1, ["2"])

    def test_kg_to_recipe_multiple_constants_round_trip(self):
        """Test multiple constants through full round-trip."""
        # First round-trip
        graph1 = get_knowledge_graph(weighted_sum.flowrep_recipe)
        reconstructed1 = kg2recipe(graph1)

        # Verify execution
        test_inputs = {"x": 1, "y": 2, "z": 3}
        expected = 10 * 1 + 20 * 2 + 30 * 3  # 140

        original_result = fr.tools.run_recipe(
            weighted_sum.flowrep_recipe, **test_inputs
        )
        reconstructed_result = fr.tools.run_recipe(reconstructed1, **test_inputs)

        original_value = original_result.output_ports["result"].value
        reconstructed_value = reconstructed_result.output_ports["result"].value

        self.assertEqual(original_value, expected)
        self.assertEqual(reconstructed_value, expected)

        # Second round-trip
        graph2 = get_knowledge_graph(reconstructed1)
        reconstructed2 = kg2recipe(graph2)

        # Verify second round-trip also works
        reconstructed2_result = fr.tools.run_recipe(reconstructed2, **test_inputs)
        reconstructed2_value = reconstructed2_result.output_ports["result"].value

        self.assertEqual(reconstructed2_value, expected)

    def test_ensure_workflow_name(self):
        graph = Graph()
        uri_to_node = kgf._uri_to_node_names(graph)
        self.assertRaises(ValueError, kgf._ensure_workflow_name, uri_to_node)
        result_1 = fr.tools.run_recipe(multiply_then_add.flowrep_recipe, x=0, y=1)
        graph = get_knowledge_graph(result_1)
        self.assertRaises(
            ValueError, kgf._ensure_workflow_name, uri_to_node, "unknown_workflow"
        )
        uri_to_node = kgf._uri_to_node_names(graph)
        self.assertEqual(
            kgf._ensure_workflow_name(uri_to_node),
            onto.BASE["Wbcedc6be_multiply_then_add"],
        )
        self.assertEqual(
            kgf._ensure_workflow_name(uri_to_node, "multiply_then_add"),
            onto.BASE["Wbcedc6be_multiply_then_add"],
        )
        uri_to_node = kgf._uri_to_node_names(graph)
        result_2 = fr.tools.run_recipe(multiply_then_add.flowrep_recipe, x=1, y=2)
        graph += get_knowledge_graph(result_2, prefix="something")
        uri_to_node = kgf._uri_to_node_names(graph)
        self.assertRaises(ValueError, kgf._ensure_workflow_name, uri_to_node)
        self.assertRaises(
            ValueError, kgf._ensure_workflow_name, uri_to_node, "multiply_then_add"
        )
        self.assertEqual(
            kgf._ensure_workflow_name(
                uri_to_node, onto.BASE["Wbcedc6be_multiply_then_add"]
            ),
            onto.BASE["Wbcedc6be_multiply_then_add"],
        )
        self.assertRaises(
            ValueError, kgf._ensure_workflow_name, uri_to_node, "unknown_workflow"
        )


if __name__ == "__main__":
    unittest.main()
