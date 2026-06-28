import unittest

import flowrep as fr

from semantikon import get_knowledge_graph, knowledge_graph_to_flowrep_recipe
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


class TestKgToFlowrep(unittest.TestCase):
    def test_round_trip_from_knowledge_graph(self):
        graph = get_knowledge_graph(my_workflow.flowrep_recipe)
        reconstructed = knowledge_graph_to_flowrep_recipe(graph)

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
            _ = knowledge_graph_to_flowrep_recipe(graph)

        reconstructed = knowledge_graph_to_flowrep_recipe(
            graph, workflow_name="multiply_by_two"
        )
        converted_result = fr.tools.run_recipe(reconstructed, x=7)
        self.assertEqual(converted_result.output_ports["result"].value, 14)

    def test_requires_t_box_information(self):
        graph = get_knowledge_graph(my_workflow.flowrep_recipe, include_t_box=False)
        with self.assertRaisesRegex(ValueError, "No workflow nodes found in graph"):
            _ = knowledge_graph_to_flowrep_recipe(graph)


if __name__ == "__main__":
    unittest.main()
