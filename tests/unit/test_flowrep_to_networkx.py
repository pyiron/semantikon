import unittest

import flowrep as fr
from rdflib import BNode

from semantikon import flowrep_to_networkx as ftn
from tests.unit.test_ontology import (
    NewSpeedData,
    my_kinetic_energy_workflow,
    passthrough_input_workflow,
    workflow_with_default_values,
)


class TestFlowrepToNetworkx(unittest.TestCase):
    def test_hash(self):
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        G = ftn.serialize_and_convert_to_networkx(wf_dict, hash_data=False)
        self.assertIsInstance(ftn._get_graph_hash(G), str)
        self.assertEqual(len(ftn._get_graph_hash(G)), 32)
        self.assertIn(
            "dtype",
            G.nodes["my_kinetic_energy_workflow-get_speed_0-inputs-distance"],
            msg="dtype should not be deleted after hashing",
        )
        self.assertEqual(
            G._get_data_node("my_kinetic_energy_workflow-get_kinetic_energy_0-inputs-velocity"),
            G._get_data_node("my_kinetic_energy_workflow-get_speed_0-outputs-speed"),
        )
        self.assertNotEqual(
            G._get_data_node("my_kinetic_energy_workflow-get_kinetic_energy_0-inputs-velocity"),
            G._get_data_node(
                "my_kinetic_energy_workflow-get_kinetic_energy_0-outputs-kinetic_energy"
            ),
        )
        wf_dict_one = my_kinetic_energy_workflow.run(distance=1.0, time=2.0, mass=3.0)
        wf_dict_two = my_kinetic_energy_workflow.run(distance=4.0, time=5.0, mass=6.0)
        G_one = ftn.serialize_and_convert_to_networkx(wf_dict_one, hash_data=True)
        G_two = ftn.serialize_and_convert_to_networkx(wf_dict_two, hash_data=True)
        self.assertEqual(
            ftn._get_graph_hash(G_one, with_global_inputs=False),
            ftn._get_graph_hash(G_two, with_global_inputs=False),
        )

        wf_dict = workflow_with_default_values.get_semantikon_dict()
        wf_dict_run = workflow_with_default_values.run(distance=2, time=1, mass=4)
        G = ftn.serialize_and_convert_to_networkx(wf_dict, hash_data=False)
        G_run = ftn.serialize_and_convert_to_networkx(wf_dict_run, hash_data=False)
        self.assertEqual(ftn._get_graph_hash(G), ftn._get_graph_hash(G_run))
        G_hash = ftn.serialize_and_convert_to_networkx(wf_dict_run, hash_data=True)
        self.assertDictEqual(
            {key.split("@")[1]: value for key, value in G_hash.get_hash_dict().items()},
            {"kinetic_energy": 8.0, "speed": 2.0},
        )
        with self.assertRaises(TypeError):
            wf_dict["inputs"]["distance"]["default"] = NewSpeedData
            G = ftn.serialize_and_convert_to_networkx(wf_dict, hash_data=True)
            ftn._get_graph_hash(G, with_global_inputs=True)
        with self.assertRaises(TypeError):
            wf_dict["inputs"]["distance"]["default"] = BNode()
            G = ftn.serialize_and_convert_to_networkx(wf_dict, hash_data=True)
            ftn._get_graph_hash(G, with_global_inputs=True)

    def test_hash_with_value(self):
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        G = ftn.serialize_and_convert_to_networkx(wf_dict, hash_data=False)
        wf_dict = my_kinetic_energy_workflow.run(distance=1, time=2, mass=3)
        G_run = ftn.serialize_and_convert_to_networkx(wf_dict, hash_data=False)
        self.assertEqual(
            ftn._get_graph_hash(G, with_global_inputs=False),
            ftn._get_graph_hash(G_run, with_global_inputs=False),
        )
        self.assertNotEqual(
            ftn._get_graph_hash(G_run, with_global_inputs=False),
            ftn._get_graph_hash(G_run, with_global_inputs=True),
        )

    def test_infer_workflow_label_without_reference(self):
        recipe = fr.schemas.WorkflowRecipe(
            inputs=["x"],
            outputs=["y"],
            nodes={},
            input_edges={},
            edges={},
            output_edges={
                fr.schemas.OutputTarget(port="y"): fr.schemas.InputSource(port="x")
            },
        )
        self.assertEqual(ftn._infer_workflow_label(recipe), "")

    def test_serialize_workflow_recipe_with_input_passthrough(self):
        self.assertTrue(
            any(
                isinstance(source, fr.schemas.InputSource)
                for source in passthrough_input_workflow.flowrep_recipe.output_edges.values()
            )
        )
        G = ftn.serialize_and_convert_to_networkx(
            passthrough_input_workflow.flowrep_recipe,
            hash_data=False,
        )
        self.assertIn(
            (
                "passthrough_input_workflow-inputs-x",
                "passthrough_input_workflow-outputs-x",
            ),
            G.edges,
        )

    def test_hashing_skips_nodes_without_function_metadata(self):
        recipe = fr.schemas.WorkflowRecipe(
            inputs=["x"],
            outputs=["y"],
            nodes={},
            input_edges={},
            edges={},
            output_edges={
                fr.schemas.OutputTarget(port="y"): fr.schemas.InputSource(port="x")
            },
        )
        data = fr.schemas.DagData.from_recipe(recipe)
        data.input_ports["x"].value = 1.0
        G = ftn._workflow_to_networkx(data)
        hashed = ftn._get_hashed_node_dict_from_graph(G)
        self.assertEqual(hashed, {})


if __name__ == "__main__":
    unittest.main()
