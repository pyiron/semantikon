import unittest
from pathlib import Path

from semantikon import cwl


class TestOntology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.static_dir = Path(__file__).parent.parent / "static"

    def test_ontology(self):
        g = cwl.get_knowledge_graph(
            self.static_dir / "cwl" / "kinetic_energy_workflow.cwl"
        )


if __name__ == "__main__":
    unittest.main()
