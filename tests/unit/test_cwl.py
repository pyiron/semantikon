import os
import unittest
from pathlib import Path

try:
    from semantikon import cwl
except ImportError:
    cwl = None

from semantikon.flowrep_to_networkx import Input, Node, Output
from semantikon.ontology import SemantikonDiGraph


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
            Node(name="get_speed", parent=Node("kinetic_energy_workflow")), g.nodes
        )
        self.assertIn(
            Node(name="get_kinetic_energy", parent=Node("kinetic_energy_workflow")),
            g.nodes,
        )

    def test_input_binding_position(self):
        g = cwl.serialize_and_convert_to_networkx(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertEqual(
            g.nodes[
                Input(
                    node=Node(parent=Node("kinetic_energy_workflow"), name="get_speed"),
                    port="distance",
                )
            ]["position"],
            1,
        )
        self.assertEqual(
            g.nodes[
                Input(
                    node=Node(parent=Node("kinetic_energy_workflow"), name="get_speed"),
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
                    node=Node(parent=Node("kinetic_energy_workflow"), name="get_speed"),
                    port="distance",
                ),
            ),
            g.edges,
        )
        self.assertIn(
            (
                Input(
                    node=Node(parent=Node("kinetic_energy_workflow"), name="get_speed"),
                    port="distance",
                ),
                Node(name="get_speed", parent=Node("kinetic_energy_workflow")),
            ),
            g.edges,
        )
        # speed flows from get_speed output -> get_kinetic_energy input
        self.assertIn(
            (
                Output(
                    node=Node(parent=Node("kinetic_energy_workflow"), name="get_speed"),
                    port="speed",
                ),
                Input(
                    node=Node(
                        parent=Node("kinetic_energy_workflow"),
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
                        parent=Node("kinetic_energy_workflow"),
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


if __name__ == "__main__":
    unittest.main()
