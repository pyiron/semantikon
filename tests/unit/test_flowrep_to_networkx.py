import unittest
from dataclasses import dataclass, field
from typing import Annotated

import flowrep as fr
from rdflib import BNode, Namespace

from semantikon import flowrep_to_networkx as ftn
from semantikon.metadata import meta

EX: Namespace = Namespace("http://example.org/")
PMD: Namespace = Namespace("https://w3id.org/pmd/co/PMD_")


@dataclass
class NewSpeedData:
    distance: Annotated[float, {"uri": PMD["0040001"], "units": "meter"}]
    time: float = field(metadata={"units": "second"})


def get_speed(
    distance: Annotated[
        float, {"uri": PMD["0040001"], "units": "meter", "label": "Distance"}
    ],
    time: Annotated[float, {"units": "second"}],
) -> Annotated[float, {"units": "meter/second", "uri": EX.Velocity, "label": "speed"}]:
    """some random docstring"""
    speed = distance / time
    return speed


@meta(uri=EX.get_kinetic_energy)
def get_kinetic_energy(
    mass: Annotated[float, {"uri": PMD["0020133"], "units": "kilogram"}],
    velocity: Annotated[float, {"units": "meter/second", "uri": EX.Velocity}],
) -> Annotated[
    float, {"uri": PMD["0020142"], "units": "joule", "label": "kinetic_energy"}
]:
    return 0.5 * mass * velocity**2


@fr.workflow
def my_kinetic_energy_workflow(
    distance: Annotated[float, {"uri": PMD["0040001"]}], time, mass
):
    speed = get_speed(distance, time)
    kinetic_energy = get_kinetic_energy(mass, speed)
    return kinetic_energy


@fr.workflow
def passthrough_input_workflow(x):
    return x


@fr.workflow
def workflow_with_default_values(distance=2, time=1, mass=4):
    speed = get_speed(distance, time)
    kinetic_energy = get_kinetic_energy(mass, speed)
    return kinetic_energy


@fr.workflow
def double_via_constant(x):
    doubled = fr.std.mul(2, x)
    return doubled


class TestFlowrepToNetworkx(unittest.TestCase):
    def test_namespace_fragments_are_base_agnostic(self):
        wf_dict = my_kinetic_energy_workflow.flowrep_recipe
        G = ftn.serialize_and_convert_to_networkx(wf_dict, hash_data=False)

        self.assertIsInstance(G.t_ns, str)
        self.assertIsInstance(G.a_ns, str)
        self.assertTrue(G.t_ns.endswith("_"))
        self.assertTrue(G.a_ns.endswith("_"))
        self.assertNotIn("http://", G.t_ns)
        self.assertNotIn("http://", G.a_ns)

        G_prefixed = ftn.serialize_and_convert_to_networkx(
            wf_dict,
            hash_data=False,
            prefix="custom",
        )
        self.assertEqual(G_prefixed.t_ns, "custom_")

    def test_hash(self):
        wf_dict = my_kinetic_energy_workflow.flowrep_recipe
        G = ftn.serialize_and_convert_to_networkx(wf_dict, hash_data=False)
        self.assertIsInstance(ftn._get_graph_hash(G), str)
        self.assertEqual(len(ftn._get_graph_hash(G)), 32)
        self.assertIn(
            "dtype",
            G.nodes[
                ftn.Input(
                    node=ftn.Node(
                        parent=ftn.Node("my_kinetic_energy_workflow"),
                        name="get_speed_0",
                    ),
                    port="distance",
                )
            ],
            msg="dtype should not be deleted after hashing",
        )
        self.assertEqual(
            G._get_data_node(
                ftn.Input(
                    node=ftn.Node(
                        parent=ftn.Node("my_kinetic_energy_workflow"),
                        name="get_kinetic_energy_0",
                    ),
                    port="velocity",
                )
            ),
            G._get_data_node(
                ftn.Output(
                    node=ftn.Node(
                        parent=ftn.Node("my_kinetic_energy_workflow"),
                        name="get_speed_0",
                    ),
                    port="speed",
                )
            ),
        )
        self.assertNotEqual(
            G._get_data_node(
                ftn.Input(
                    node=ftn.Node(
                        parent=ftn.Node("my_kinetic_energy_workflow"),
                        name="get_kinetic_energy_0",
                    ),
                    port="velocity",
                )
            ),
            G._get_data_node(
                ftn.Output(
                    node=ftn.Node(
                        parent=ftn.Node("my_kinetic_energy_workflow"),
                        name="get_kinetic_energy_0",
                    ),
                    port="kinetic_energy",
                )
            ),
        )
        wf_dict_one = fr.wfms.run_recipe(
            my_kinetic_energy_workflow.flowrep_recipe, distance=1.0, time=2.0, mass=3.0
        )
        wf_dict_two = fr.wfms.run_recipe(
            my_kinetic_energy_workflow.flowrep_recipe, distance=4.0, time=5.0, mass=6.0
        )
        G_one = ftn.serialize_and_convert_to_networkx(wf_dict_one, hash_data=True)
        G_two = ftn.serialize_and_convert_to_networkx(wf_dict_two, hash_data=True)
        self.assertEqual(
            ftn._get_graph_hash(G_one, with_global_inputs=False),
            ftn._get_graph_hash(G_two, with_global_inputs=False),
        )

        wf_dict = workflow_with_default_values.flowrep_recipe
        wf_dict_run = fr.wfms.run_recipe(
            workflow_with_default_values.flowrep_recipe, distance=2, time=1, mass=4
        )
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
        wf_dict = my_kinetic_energy_workflow.flowrep_recipe
        G = ftn.serialize_and_convert_to_networkx(wf_dict, hash_data=False)
        wf_dict = fr.wfms.run_recipe(
            my_kinetic_energy_workflow.flowrep_recipe, distance=1, time=2, mass=3
        )
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
                ftn.Input(node=ftn.Node("passthrough_input_workflow"), port="x"),
                ftn.Output(node=ftn.Node("passthrough_input_workflow"), port="x"),
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

    def test_add_node_validates_semantikon_metadata(self):
        G = ftn.SemantikonDiGraph()
        G.add_node(ftn.Node("n"), type="atomic")
        self.assertEqual(G.nodes[ftn.Node("n")]["type"], "atomic")

        node_in = ftn.Input(node=ftn.Node("in"), port="x")
        G.add_node(node_in, position=0, value=None)
        self.assertIn("value", G.nodes[node_in])
        self.assertIsNone(G.nodes[node_in]["value"])

        node_out = ftn.Output(node=ftn.Node("out"), port="y")
        G.add_node(node_out, position=1, default=3.14)
        self.assertEqual(G.nodes[node_out]["default"], 3.14)
        self.assertNotIn("value", G.nodes[node_out])

    def test_add_node_rejects_invalid_semantikon_metadata(self):
        G = ftn.SemantikonDiGraph()
        with self.assertRaises(ValueError):
            G.add_node(ftn.Node("n"), type="invalid")

    def test_add_nodes_from_validates_semantikon_metadata(self):
        G = ftn.SemantikonDiGraph()
        G.add_nodes_from(
            [ftn.Input(ftn.Node("n1"), port="x"), ftn.Input(ftn.Node("n2"), port="x")],
            position=0,
            value=None,
        )

        for n in ("n1", "n2"):
            node = ftn.Input(ftn.Node(n), port="x")
            self.assertEqual(G.nodes[node]["position"], 0)
            self.assertIn("value", G.nodes[node])
            self.assertIsNone(G.nodes[node]["value"])

    def test_add_nodes_from_validates_per_node_semantikon_metadata_without_shared_attrs(
        self,
    ):
        G = ftn.SemantikonDiGraph()
        n_1 = ftn.Input(ftn.Node("n1"), port="x")
        n_2 = ftn.Input(ftn.Node("n2"), port="y")
        G.add_nodes_from(
            [
                (n_1, {"position": 0, "value": 1}),
                (n_2, {"position": 1}),
            ]
        )

        self.assertEqual(G.nodes[n_1]["position"], 0)
        self.assertEqual(G.nodes[n_1]["value"], 1)

        self.assertEqual(G.nodes[n_2]["position"], 1)
        self.assertNotIn("value", G.nodes[n_2])

    def test_add_nodes_from_preserves_default_on_inputs(self):
        G = ftn.SemantikonDiGraph()
        n_1 = ftn.Input(ftn.Node("n1"), port="x")
        G.add_nodes_from([(n_1, {"position": 0, "default": 7})])

        self.assertEqual(G.nodes[n_1]["position"], 0)
        self.assertIn("default", G.nodes[n_1])
        self.assertEqual(G.nodes[n_1]["default"], 7)

    def test_add_nodes_from_merges_per_node_semantikon_metadata(self):
        G = ftn.SemantikonDiGraph()
        n_1 = ftn.Input(ftn.Node("n1"), port="x")
        n_2 = ftn.Input(ftn.Node("n2"), port="y")
        G.add_nodes_from(
            [
                (n_1, {"position": 0, "value": 1}),
                (n_2, {"position": 1}),
            ],
            value=None,
            dtype="float",
        )

        self.assertEqual(G.nodes[n_1]["position"], 0)
        self.assertEqual(G.nodes[n_1]["dtype"], "float")
        self.assertEqual(G.nodes[n_1]["value"], 1)

        self.assertEqual(G.nodes[n_2]["position"], 1)
        self.assertEqual(G.nodes[n_2]["dtype"], "float")
        self.assertIsNone(G.nodes[n_2]["value"])

    def test_add_nodes_from_rejects_invalid_semantikon_metadata(self):
        G = ftn.SemantikonDiGraph()
        with self.assertRaises(ValueError):
            G.add_nodes_from([ftn.Node("n")], type="invalid")

    def test_add_nodes_from_rejects_invalid_per_node_semantikon_metadata(self):
        G = ftn.SemantikonDiGraph()
        with self.assertRaises(ValueError):
            G.add_nodes_from([(ftn.Node("n"), {"type": "invalid"})])

    def test_constant(self):
        G = ftn.serialize_and_convert_to_networkx(double_via_constant.flowrep_recipe)
        self.assertEqual(
            G.nodes[
                ftn.Input(
                    node=ftn.Node(parent=ftn.Node("double_via_constant"), name="mul_0"),
                    port="a",
                )
            ]["constant_value"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
