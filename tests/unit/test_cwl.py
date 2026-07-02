import os
import unittest
from pathlib import Path

from semantikon import cwl
from semantikon.ontology import SemantikonDiGraph


class TestCWL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.static_dir = Path(__file__).parent.parent / "static"

    @unittest.skipIf(os.name == "nt", "Skipping test on Windows")
    def test_returns_semantikon_digraph(self):
        g = cwl.get_knowledge_graph(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertIsInstance(g, SemantikonDiGraph)

    @unittest.skipIf(os.name == "nt", "Skipping test on Windows")
    def test_graph_prefix(self):
        g = cwl.get_knowledge_graph(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertEqual(g.graph["prefix"], "kinetic_energy_workflow")

    @unittest.skipIf(os.name == "nt", "Skipping test on Windows")
    def test_workflow_input_nodes(self):
        g = cwl.get_knowledge_graph(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        expected_inputs = {
            "kinetic_energy_workflow-inputs-distance",
            "kinetic_energy_workflow-inputs-time",
            "kinetic_energy_workflow-inputs-mass",
        }
        self.assertTrue(expected_inputs.issubset(set(g.nodes)))

    @unittest.skipIf(os.name == "nt", "Skipping test on Windows")
    def test_workflow_output_nodes(self):
        g = cwl.get_knowledge_graph(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertIn("kinetic_energy_workflow-outputs-kinetic_energy", g.nodes)

    @unittest.skipIf(os.name == "nt", "Skipping test on Windows")
    def test_step_nodes(self):
        g = cwl.get_knowledge_graph(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertIn("kinetic_energy_workflow-get_speed", g.nodes)
        self.assertIn("kinetic_energy_workflow-get_kinetic_energy", g.nodes)

    @unittest.skipIf(os.name == "nt", "Skipping test on Windows")
    def test_node_step_attributes(self):
        g = cwl.get_knowledge_graph(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertEqual(
            g.nodes["kinetic_energy_workflow-inputs-distance"]["step"], "inputs"
        )
        self.assertEqual(
            g.nodes["kinetic_energy_workflow-outputs-kinetic_energy"]["step"], "outputs"
        )
        self.assertEqual(g.nodes["kinetic_energy_workflow-get_speed"]["step"], "node")

    @unittest.skipIf(os.name == "nt", "Skipping test on Windows")
    def test_input_binding_position(self):
        g = cwl.get_knowledge_graph(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        self.assertEqual(
            g.nodes["kinetic_energy_workflow-get_speed-inputs-distance"]["position"], 1
        )
        self.assertEqual(
            g.nodes["kinetic_energy_workflow-get_speed-inputs-time"]["position"], 2
        )

    @unittest.skipIf(os.name == "nt", "Skipping test on Windows")
    def test_data_flow_edges(self):
        g = cwl.get_knowledge_graph(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )
        # distance flows from workflow input -> get_speed input -> get_speed step
        self.assertIn(
            (
                "kinetic_energy_workflow-inputs-distance",
                "kinetic_energy_workflow-get_speed-inputs-distance",
            ),
            g.edges,
        )
        self.assertIn(
            (
                "kinetic_energy_workflow-get_speed-inputs-distance",
                "kinetic_energy_workflow-get_speed",
            ),
            g.edges,
        )
        # speed flows from get_speed output -> get_kinetic_energy input
        self.assertIn(
            (
                "kinetic_energy_workflow-get_speed-outputs-speed",
                "kinetic_energy_workflow-get_kinetic_energy-inputs-velocity",
            ),
            g.edges,
        )
        # kinetic_energy flows from step output -> workflow output
        self.assertIn(
            (
                "kinetic_energy_workflow-get_kinetic_energy-outputs-kinetic_energy",
                "kinetic_energy_workflow-outputs-kinetic_energy",
            ),
            g.edges,
        )

    @unittest.skipIf(os.name == "nt", "Skipping test on Windows")
    def test_get_name(self):
        self.assertEqual(
            cwl._get_name("file:///path/to/file.cwl#local_name"), "local_name"
        )
        self.assertEqual(cwl._get_name("no_fragment"), "no_fragment")
        self.assertEqual(cwl._get_name("a#b#c"), "c")


if __name__ == "__main__":
    unittest.main()
