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


def arange(min_value, max_value):
    result = range(min_value, max_value)
    return result


def sub(a, b):
    result = a - b
    return result


def mean(values):
    result = sum(values) / len(values)
    return result


@fr.workflow
def shift_all(values, x):
    result = []
    for v in values:
        a = sub(v, x)
        result.append(a)
    return result


@fr.workflow
def mean_of_loop(min_value=0, max_value=3, x=2):
    r = arange(min_value=min_value, max_value=max_value)
    result = []
    for v in r:
        a = sub(v, x)
        result.append(a)
    mean_value = mean(result)
    return mean_value


class TestFlowrepToNetworkx(unittest.TestCase):
    def test_namespace_fragments_are_base_agnostic(self):
        wf_dict = my_kinetic_energy_workflow.flowrep_recipe
        G = ftn.serialize_instance_to_networkx(wf_dict, hash_data=False)

        self.assertIsInstance(G.t_ns, str)
        self.assertIsInstance(G.a_ns, str)
        self.assertTrue(G.t_ns.endswith("_"))
        self.assertTrue(G.a_ns.endswith("_"))
        self.assertNotIn("http://", G.t_ns)
        self.assertNotIn("http://", G.a_ns)

        G_prefixed = ftn.serialize_and_convert_to_networkx(
            wf_dict,
            prefix="custom",
        )
        self.assertEqual(G_prefixed.t_ns, "custom_")

    def test_hash(self):
        wf_dict = my_kinetic_energy_workflow.flowrep_recipe
        G = ftn.serialize_and_convert_to_networkx(wf_dict)
        self.assertIsInstance(ftn._get_graph_hash(G), str)
        self.assertEqual(len(ftn._get_graph_hash(G)), 32)
        self.assertIn(
            "dtype",
            G.nodes[
                ftn.Input(
                    node=ftn.Node(
                        owner=ftn.Node("my_kinetic_energy_workflow"),
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
                        owner=ftn.Node("my_kinetic_energy_workflow"),
                        name="get_kinetic_energy_0",
                    ),
                    port="velocity",
                )
            ),
            G._get_data_node(
                ftn.Output(
                    node=ftn.Node(
                        owner=ftn.Node("my_kinetic_energy_workflow"),
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
                        owner=ftn.Node("my_kinetic_energy_workflow"),
                        name="get_kinetic_energy_0",
                    ),
                    port="velocity",
                )
            ),
            G._get_data_node(
                ftn.Output(
                    node=ftn.Node(
                        owner=ftn.Node("my_kinetic_energy_workflow"),
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
        G_one = ftn.serialize_instance_to_networkx(wf_dict_one, hash_data=True)
        G_two = ftn.serialize_instance_to_networkx(wf_dict_two, hash_data=True)
        self.assertEqual(
            ftn._get_graph_hash(G_one, with_global_inputs=False),
            ftn._get_graph_hash(G_two, with_global_inputs=False),
        )

        wf_dict = workflow_with_default_values.flowrep_recipe
        wf_dict_run = fr.wfms.run_recipe(
            workflow_with_default_values.flowrep_recipe, distance=2, time=1, mass=4
        )
        G = ftn.serialize_instance_to_networkx(wf_dict, hash_data=False)
        G_run = ftn.serialize_instance_to_networkx(wf_dict_run, hash_data=False)
        self.assertEqual(ftn._get_graph_hash(G), ftn._get_graph_hash(G_run))
        G_hash = ftn.serialize_instance_to_networkx(wf_dict_run, hash_data=True)
        self.assertDictEqual(
            {key.split("@")[1]: value for key, value in G_hash.get_hash_dict().items()},
            {"kinetic_energy": 8.0, "speed": 2.0},
        )
        with self.assertRaises(TypeError):
            wf_dict["inputs"]["distance"]["default"] = NewSpeedData
            G = ftn.serialize_instance_to_networkx(wf_dict, hash_data=True)
            ftn._get_graph_hash(G, with_global_inputs=True)
        with self.assertRaises(TypeError):
            wf_dict["inputs"]["distance"]["default"] = BNode()
            G = ftn.serialize_instance_to_networkx(wf_dict, hash_data=True)
            ftn._get_graph_hash(G, with_global_inputs=True)

    def test_hash_with_value(self):
        wf_dict = my_kinetic_energy_workflow.flowrep_recipe
        G = ftn.serialize_instance_to_networkx(wf_dict, hash_data=False)
        wf_dict = fr.wfms.run_recipe(
            my_kinetic_energy_workflow.flowrep_recipe, distance=1, time=2, mass=3
        )
        G_run = ftn.serialize_instance_to_networkx(wf_dict, hash_data=False)
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
        self.assertEqual(ftn._infer_workflow_label(recipe), "workflow")

    def test_infer_workflow_label_with_explicit_label(self):
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
        self.assertEqual(
            ftn._infer_workflow_label(recipe, label="my_label"), "my_label"
        )

    def test_graph_name_uses_prefix(self):
        wf_dict = my_kinetic_energy_workflow.flowrep_recipe
        G = ftn.serialize_and_convert_to_networkx(
            wf_dict, hash_data=False, prefix="my_prefix"
        )
        self.assertEqual(G.name, "my_prefix")

    def test_prefix_does_not_override_named_workflow_node(self):
        wf_dict = my_kinetic_energy_workflow.flowrep_recipe
        G = ftn.serialize_and_convert_to_networkx(
            wf_dict, hash_data=False, prefix="my_prefix"
        )
        self.assertIn(ftn.Node("my_kinetic_energy_workflow"), G.nodes)

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
        A, _ = ftn._workflow_to_instance_graph(data)
        hashed = ftn._get_hashed_node_dict_from_graph(A)
        self.assertEqual(hashed, {})

    def test_recipe_graph_has_no_values(self):
        G = ftn.SemantikonRecipeGraph()
        node_in = ftn.Input(node=ftn.Node("in"), port="x")
        with self.assertRaises(ValueError):
            G.add_node(node_in, position=0, value=3)
        node_out = ftn.Output(node=ftn.Node("out"), port="y")
        G.add_node(node_out, position=1, default=3.14)
        self.assertEqual(G.nodes[node_out]["default"], 3.14)

    def test_instance_graph_references_recipe(self):
        wf = my_kinetic_energy_workflow.flowrep_recipe
        data = fr.wfms.run_recipe(wf, distance=1.0, time=2.0, mass=3.0)
        A = ftn.serialize_instance_to_networkx(data, hash_data=True)
        R = A.recipe_graph
        self.assertIsInstance(R, ftn.SemantikonRecipeGraph)
        self.assertFalse(any("value" in d for _, d in R.nodes.data()))
        for node, d in A.nodes.data():
            self.assertIn(d["recipe"], R.nodes)
            if isinstance(node, ftn.ANode):
                self.assertIsInstance(node, ftn.ANode)
                self.assertEqual(d["status"], "pending")
            else:
                self.assertIsInstance(node, (ftn.AInput, ftn.AOutput))
        inputs = {
            n.port: d["value"]
            for n, d in A.nodes.data()
            if isinstance(n, ftn.AInput) and n.node.owner is None
        }
        self.assertEqual(inputs, {"distance": 1.0, "time": 2.0, "mass": 3.0})

    def test_instance_graph_rejects_recipe_outside_graph(self):
        recipe_graph = ftn.SemantikonRecipeGraph()
        recipe_graph.add_node(ftn.Node("recipe"))
        graph = ftn.SemantikonInstanceGraph(recipe_graph=recipe_graph)

        with self.assertRaises(ValueError):
            graph.add_node(ftn.ANode("instance"), recipe=ftn.Node("other"))

    def test_instance_graph_rejects_mismatched_recipe_types(self):
        recipe_graph = ftn.SemantikonRecipeGraph()
        recipe_node = ftn.Node("recipe")
        recipe_input = ftn.Input(recipe_node, "x")
        recipe_output = ftn.Output(recipe_node, "y")
        recipe_graph.add_node(recipe_node)
        recipe_graph.add_node(recipe_input, position=0)
        recipe_graph.add_node(recipe_output, position=0)
        graph = ftn.SemantikonInstanceGraph(recipe_graph=recipe_graph)

        invalid_references = [
            (ftn.ANode("node_instance"), recipe_input),
            (ftn.AInput(ftn.ANode("input_instance"), "x"), recipe_output),
            (ftn.AOutput(ftn.ANode("output_instance"), "y"), recipe_input),
            (ftn.ANode("object_instance"), []),
        ]
        for instance, recipe in invalid_references:
            with (
                self.subTest(instance=instance, recipe=recipe),
                self.assertRaises(ValueError),
            ):
                graph.add_node(instance, recipe=recipe)

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
                (n_1, {"position": 0, "default": 1}),
                (n_2, {"position": 1}),
            ],
            dtype="float",
        )

        self.assertEqual(G.nodes[n_1]["position"], 0)
        self.assertEqual(G.nodes[n_1]["dtype"], "float")
        self.assertEqual(G.nodes[n_1]["default"], 1)

        self.assertEqual(G.nodes[n_2]["position"], 1)
        self.assertEqual(G.nodes[n_2]["dtype"], "float")
        self.assertNotIn("default", G.nodes[n_2])

    def test_constant(self):
        G = ftn.serialize_and_convert_to_networkx(double_via_constant.flowrep_recipe)
        inp = ftn.Input(
            node=ftn.Node(owner=ftn.Node("double_via_constant"), name="mul_0"),
            port="a",
        )
        self.assertIn(inp, G.nodes)
        self.assertEqual(G.nodes[inp]["constant_value"], 2)

    def test_for_each_recipe_graph(self):
        G = ftn.serialize_and_convert_to_networkx(mean_of_loop.flowrep_recipe)
        types = {n.name: G.get_type(n) for n in G.nodes if isinstance(n, ftn.Node)}
        self.assertEqual(types["for_each_0"], "for_each")
        self.assertEqual(types["distributor_r"], "distributor")
        self.assertEqual(types["collector_result"], "collector")
        self.assertEqual(types["body"], "workflow")
        self.assertEqual(
            sum(isinstance(n, ftn.Node) and n.name == "sub_0" for n in G.nodes), 1
        )
        for_each = ftn.Node("for_each_0", ftn.Node("mean_of_loop"))
        dist = ftn.Node("distributor_r", for_each)
        coll = ftn.Node("collector_result", for_each)
        body = ftn.Node("body", for_each)
        for edge in [
            (ftn.Input(for_each, "r"), ftn.Input(dist, "items")),
            (ftn.Output(dist, "item"), ftn.Input(body, "v")),
            (ftn.Input(for_each, "x"), ftn.Input(body, "x")),
            (ftn.Output(body, "a"), ftn.Input(coll, "item")),
            (ftn.Output(coll, "items"), ftn.Output(for_each, "result")),
        ]:
            self.assertIn(edge, G.edges)

    def test_for_each_instance_graph(self):
        recipe = mean_of_loop.flowrep_recipe

        def count(A, name, cls=ftn.ANode):
            return sum(isinstance(n, cls) and n.name == name for n in A.nodes)

        pending = ftn.serialize_instance_to_networkx(recipe)
        self.assertEqual(count(pending, "body"), 1)
        self.assertEqual(count(pending, "sub_0"), 1)

        executed = ftn.serialize_instance_to_networkx(fr.wfms.run_recipe(recipe))
        for A in (pending, executed):
            self.assertTrue(all("recipe" in d for _, d in A.nodes.data()))
            self.assertEqual(A.recipe_graph.nodes, pending.recipe_graph.nodes)
        self.assertEqual(count(executed, "body"), 3)
        self.assertEqual(count(executed, "sub_0"), 3)
        self.assertEqual(count(executed, "distributor_r"), 1)
        self.assertEqual(
            sorted(
                n.counter
                for n in executed.nodes
                if isinstance(n, ftn.AOutput) and n.port == "item"
            ),
            [0, 1, 2],
        )
        collected = {
            n.counter: d["value"]
            for n, d in executed.nodes.data()
            if isinstance(n, ftn.AInput) and n.port == "item"
        }
        self.assertEqual(collected, {0: -2, 1: -1, 2: 0})
        result = next(
            d["value"]
            for n, d in executed.nodes.data()
            if isinstance(n, ftn.AOutput)
            and n.port == "result"
            and n.node.name == "for_each_0"
        )
        self.assertEqual(result, [-2, -1, 0])

    def test_for_each_without_iterations(self):
        data = fr.wfms.run_recipe(shift_all.flowrep_recipe, values=[], x=1)
        A = ftn.serialize_instance_to_networkx(data, hash_data=False)
        self.assertEqual(
            sum(isinstance(n, ftn.ANode) and n.name == "body" for n in A), 0
        )
        self.assertIn(
            ftn.Node("body", ftn.Node("for_each_0", ftn.Node("shift_all"))),
            A.recipe_graph.nodes,
        )


if __name__ == "__main__":
    unittest.main()
