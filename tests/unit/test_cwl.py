import os
import tempfile
import unittest
from pathlib import Path
from typing import Annotated

try:
    from semantikon import cwl
except ImportError:
    cwl = None

from semantikon.flowrep_to_networkx import Input, Node, Output
from semantikon.ontology import SemantikonDiGraph, function_to_knowledge_graph


def get_speed(
    distance: Annotated[float, {"units": "meter"}],
    time: float = 2.0,
) -> Annotated[float, {"units": "meter/second"}]:
    """compute speed"""
    return distance / time


@unittest.skipIf(
    os.name == "nt" or cwl is None,
    "Skipping CWL tests (Windows or optional CWL dependencies not installed)",
)
class TestCWL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.static_dir = Path(__file__).parent.parent / "static"

    def test_returns_semantikon_digraph(self):
        g = cwl.serialize_and_convert_to_networkx(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertIsInstance(g, SemantikonDiGraph)

    def test_graph_prefix(self):
        g = cwl.serialize_and_convert_to_networkx(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertEqual(g.graph["prefix"], "kinetic_energy_workflow")

    def test_workflow_input_nodes(self):
        g = cwl.serialize_and_convert_to_networkx(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        expected_inputs = {
            Input(node=Node("kinetic_energy_workflow"), port="distance"),
            Input(node=Node("kinetic_energy_workflow"), port="time"),
            Input(node=Node("kinetic_energy_workflow"), port="mass"),
        }
        self.assertTrue(expected_inputs.issubset(set(g.nodes)))

    def test_workflow_output_nodes(self):
        g = cwl.serialize_and_convert_to_networkx(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertIn(
            Output(node=Node("kinetic_energy_workflow"), port="kinetic_energy"), g.nodes
        )

    def test_step_nodes(self):
        g = cwl.serialize_and_convert_to_networkx(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertIn(
            Node(name="get_speed", owner=Node("kinetic_energy_workflow")), g.nodes
        )
        self.assertIn(
            Node(name="get_kinetic_energy", owner=Node("kinetic_energy_workflow")),
            g.nodes,
        )

    def test_input_binding_position(self):
        g = cwl.serialize_and_convert_to_networkx(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertEqual(
            g.nodes[
                Input(
                    node=Node(owner=Node("kinetic_energy_workflow"), name="get_speed"),
                    port="distance",
                )
            ]["position"],
            1,
        )
        self.assertEqual(
            g.nodes[
                Input(
                    node=Node(owner=Node("kinetic_energy_workflow"), name="get_speed"),
                    port="time",
                )
            ]["position"],
            2,
        )

    def test_data_flow_edges(self):
        g = cwl.serialize_and_convert_to_networkx(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        # distance flows from workflow input -> get_speed input -> get_speed step
        self.assertIn(
            (
                Input(node=Node("kinetic_energy_workflow"), port="distance"),
                Input(
                    node=Node(owner=Node("kinetic_energy_workflow"), name="get_speed"),
                    port="distance",
                ),
            ),
            g.edges,
        )
        self.assertIn(
            (
                Input(
                    node=Node(owner=Node("kinetic_energy_workflow"), name="get_speed"),
                    port="distance",
                ),
                Node(name="get_speed", owner=Node("kinetic_energy_workflow")),
            ),
            g.edges,
        )
        # speed flows from get_speed output -> get_kinetic_energy input
        self.assertIn(
            (
                Output(
                    node=Node(owner=Node("kinetic_energy_workflow"), name="get_speed"),
                    port="speed",
                ),
                Input(
                    node=Node(
                        owner=Node("kinetic_energy_workflow"),
                        name="get_kinetic_energy",
                    ),
                    port="velocity",
                ),
            ),
            g.edges,
        )
        # kinetic_energy flows from step output -> workflow output
        self.assertIn(
            (
                Output(
                    node=Node(
                        owner=Node("kinetic_energy_workflow"),
                        name="get_kinetic_energy",
                    ),
                    port="kinetic_energy",
                ),
                Output(node=Node("kinetic_energy_workflow"), port="kinetic_energy"),
            ),
            g.edges,
        )

    def test_get_name(self):
        self.assertEqual(
            cwl._get_name("file:///path/to/file.cwl#local_name"), "local_name"
        )
        self.assertEqual(cwl._get_name("no_fragment"), "no_fragment")
        self.assertEqual(cwl._get_name("a#b#c"), "c")

    def test_knowledge_graph_to_cwl(self):
        g = function_to_knowledge_graph(get_speed)
        tool = cwl.knowledge_graph_to_cwl(g)
        self.assertEqual(
            tool.id,
            "get_speed_b23355f2e639d541b86c1c53ab0559e2cc7f87d699788238c12a2276719ad0a3",
        )
        self.assertEqual(tool.doc, "compute speed")
        self.assertEqual([i.id for i in tool.inputs], ["distance", "time"])
        self.assertEqual(
            [i.position for i in [t.inputBinding for t in tool.inputs]], [0, 1]
        )
        self.assertEqual(tool.inputs[1].default, 2.0)
        self.assertEqual(tool.inputs[1].type_, "double")
        self.assertEqual(tool.inputs[0].type_, "Any")
        self.assertEqual([o.id for o in tool.outputs], ["output_0"])

    def test_knowledge_graph_to_cwl_requires_unambiguous_f_node(self):
        from rdflib import Graph

        with self.assertRaises(ValueError):
            cwl.knowledge_graph_to_cwl(Graph())

    def test_knowledge_graph_to_cwl_roundtrip(self):
        g = function_to_knowledge_graph(get_speed)
        tool = cwl.knowledge_graph_to_cwl(g)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = cwl.save_cwl_file(tool, Path(tmp_dir) / "get_speed.cwl")
            reloaded = cwl.serialize_and_convert_to_networkx(path)
        print(reloaded.nodes)
        expected_inputs = {
            Input(
                node=Node(
                    "get_speed#get_speed_b23355f2e639d541b86c1c53ab0559e2cc7f87d699788238c12a2276719ad0a3"
                ),
                port="get_speed_b23355f2e639d541b86c1c53ab0559e2cc7f87d699788238c12a2276719ad0a3/distance",
            ),
            Input(
                node=Node(
                    "get_speed#get_speed_b23355f2e639d541b86c1c53ab0559e2cc7f87d699788238c12a2276719ad0a3"
                ),
                port="get_speed_b23355f2e639d541b86c1c53ab0559e2cc7f87d699788238c12a2276719ad0a3/time",
            ),
        }
        self.assertTrue(expected_inputs.issubset(set(reloaded.nodes)))
        self.assertIn(
            Output(
                node=Node(
                    "get_speed#get_speed_b23355f2e639d541b86c1c53ab0559e2cc7f87d699788238c12a2276719ad0a3"
                ),
                port="get_speed_b23355f2e639d541b86c1c53ab0559e2cc7f87d699788238c12a2276719ad0a3/output_0",
            ),
            reloaded.nodes,
        )


if __name__ == "__main__":
    unittest.main()
