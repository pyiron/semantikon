import unittest
from typing import Annotated

from rdflib import OWL, RDF, RDFS, SH, Graph, Literal, Namespace

from semantikon import analysis as asis
from semantikon import ontology as onto
from semantikon.metadata import meta
from semantikon.workflow import workflow


EX: Namespace = Namespace("http://example.org/")
PMD: Namespace = Namespace("https://w3id.org/pmd/co/PMD_")


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


@workflow
def my_kinetic_energy_workflow(
    distance: Annotated[float, {"uri": PMD["0040001"]}], time, mass
):
    speed = get_speed(distance, time)
    kinetic_energy = get_kinetic_energy(mass, speed)
    return kinetic_energy


class TestAnalysis(unittest.TestCase):
    def test_my_kinetic_energy_workflow_graph(self):
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        g = onto.get_knowledge_graph(wf_dict, prefix="T")

        with self.subTest("workflow instance exists"):
            uri = asis.label_to_uri(g, "my_kinetic_energy_workflow")[0]
            workflows = list(g.subjects(RDF.type, uri))
            self.assertEqual(len(workflows), 1)

        wf = workflows[0]

        with self.subTest("workflow has both function executions as parts"):
            parts = list(g.objects(wf, onto.BFO["0000051"]))
            uri = asis.label_to_uri(
                g, "my_kinetic_energy_workflow-get_kinetic_energy_0"
            )[0]
            ke_calls = [p for p in parts if (p, RDF.type, uri) in g]
            uri = asis.label_to_uri(g, "my_kinetic_energy_workflow-get_speed_0")[0]
            speed_calls = [p for p in parts if (p, RDF.type, uri) in g]
            self.assertEqual(len(ke_calls), 1)
            self.assertEqual(len(speed_calls), 1)

        ke_call = ke_calls[0]
        speed_call = speed_calls[0]

        with self.subTest("speed computation precedes kinetic energy computation"):
            self.assertIn((speed_call, onto.BFO["0000063"], ke_call), g)

        with self.subTest("functions are linked to python callables"):
            self.assertIn(
                (
                    ke_call,
                    onto.RO["0000059"],
                    onto.BASE[
                        f"{__name__}-get_kinetic_energy-not_defined".replace(".", "-")
                    ],
                ),
                g,
            )
            self.assertIn(
                (
                    speed_call,
                    onto.RO["0000059"],
                    onto.BASE[f"{__name__}-get_speed-not_defined".replace(".", "-")],
                ),
                g,
            )

    def test_request_values(self):
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        wf_dict["inputs"]["distance"]["value"] = 1.0
        wf_dict["inputs"]["time"]["value"] = 2.0
        wf_dict["inputs"]["mass"]["value"] = 3.0
        self.assertDictEqual(wf_dict["outputs"], {"kinetic_energy": {}})
        graph = onto.get_knowledge_graph(my_kinetic_energy_workflow.run(2.0, 2.0, 3.0))
        wf_dict = asis.request_values(wf_dict, graph)
        self.assertDictEqual(
            wf_dict["outputs"], {"kinetic_energy": {}}, msg="no known inputs"
        )
        graph += onto.get_knowledge_graph(my_kinetic_energy_workflow.run(1.0, 2.0, 3.0))
        wf_dict = asis.request_values(wf_dict, graph)
        self.assertDictEqual(
            wf_dict["outputs"],
            {"kinetic_energy": {"value": 0.375}},
            msg="all inputs known because the same simulation was run before",
        )
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        wf_dict["inputs"]["distance"]["value"] = 1.0
        wf_dict["inputs"]["time"]["value"] = 2.0
        wf_dict["inputs"]["mass"]["value"] = 4.0
        wf_dict = asis.request_values(wf_dict, graph)
        self.assertDictEqual(
            wf_dict["outputs"],
            {"kinetic_energy": {}},
            msg="kinetic energy must be unknown because of unknown mass",
        )
        self.assertEqual(
            wf_dict["nodes"]["get_speed_0"]["outputs"]["speed"]["value"],
            0.5,
            msg="speed must be known because of known distance and time",
        )
        graph = onto.get_knowledge_graph(
            my_kinetic_energy_workflow.run(1.0, 2.0, 4.0), remove_data=True
        )
        wf_dict = asis.request_values(wf_dict, graph)
        self.assertDictEqual(
            wf_dict["outputs"], {"kinetic_energy": {}}, msg="data not stored"
        )

    def test_sparql_writer(self):
        wf_dict = my_kinetic_energy_workflow.run(2.0, 1.0, 4.0)
        graph = onto.get_knowledge_graph(wf_dict, prefix="T")
        comp = asis.query_io_completer(graph)
        self.assertEqual(
            dir(comp.T_my_kinetic_energy_workflow.get_speed_0.inputs),
            ["distance", "time"],
        )
        A = comp.T_my_kinetic_energy_workflow.inputs.time
        self.assertEqual(dir(A), ["query", "to_query_text"])
        B = comp.T_my_kinetic_energy_workflow.outputs.kinetic_energy
        C = comp.T_my_kinetic_energy_workflow.inputs.mass
        D = comp.T_my_kinetic_energy_workflow.inputs.distance
        self.assertListEqual(
            dir(comp.T_my_kinetic_energy_workflow),
            ["get_kinetic_energy_0", "get_speed_0", "inputs", "outputs"],
        )
        self.assertEqual((A & B).query(), [(1.0, 8.0)])
        self.assertEqual(
            (A & C & B).query(), [(1.0, 4.0, 8.0)], msg=(A & C & B).to_query_text()
        )
        self.assertEqual((A & (C & B)).query(), [(1.0, 4.0, 8.0)])
        self.assertEqual((A & C & D & B).query(), [(1.0, 4.0, 2.0, 8.0)])
        self.assertEqual(((A & C) & (D & B)).query(), [(1.0, 4.0, 2.0, 8.0)])
        self.assertEqual(A.query(), [(1.0,)])
        A_dash = A.value()  # A is now URIRef instead of _Node
        self.assertEqual((A_dash & (C & B)).query(), [(1.0, 4.0, 8.0)])
        self.assertEqual((A_dash & C & D & B).query(), [(1.0, 4.0, 2.0, 8.0)])
        self.assertEqual(((A_dash & C) & (D & B)).query(), [(1.0, 4.0, 2.0, 8.0)])
        self.assertEqual(list(graph.query(A.to_query_text()))[0][0].toPython(), 1.0)
        with self.assertRaises(AttributeError):
            _ = comp.non_existing_node
        self.assertIsInstance(comp.T_my_kinetic_energy_workflow, asis._Node)

        @workflow
        def only_get_speed_workflow(distance, time):
            speed = get_speed(distance=distance, time=time)
            return speed

        graph += onto.get_knowledge_graph(
            only_get_speed_workflow.run(3.0, 1.5), prefix="T"
        )
        comp = asis.query_io_completer(graph)
        A = comp.T_my_kinetic_energy_workflow.inputs.time
        B = comp.T_my_kinetic_energy_workflow.outputs.kinetic_energy
        C = comp.T_my_kinetic_energy_workflow.inputs.mass
        self.assertEqual((A & B).query(), [(1.0, 8.0)])
        self.assertEqual((A & C & B).query(), [(1.0, 4.0, 8.0)])
        self.assertEqual(A.query(), [(1.0,)])
        self.assertListEqual(
            dir(comp), ["T_my_kinetic_energy_workflow", "T_only_get_speed_workflow"]
        )
        E = comp.T_only_get_speed_workflow.inputs.distance
        with self.assertRaises(ValueError) as context:
            _ = (A & E).query()
        self.assertEqual(str(context.exception), "No common head node found")
        self.assertEqual(E.query(), [(3.0,)])
        graph = onto.get_knowledge_graph(wf_dict, prefix="T", remove_data=True)
        comp = asis.query_io_completer(graph)
        A = comp.T_my_kinetic_energy_workflow.inputs.time
        B = comp.T_my_kinetic_energy_workflow.outputs.kinetic_energy
        self.assertListEqual((A & B).query(), [])
        data = (A & B).query(fallback_to_hash=True)
        self.assertEqual(data[0][0], 1.0)
        self.assertIsInstance(data[0][1], str)
        self.assertIsInstance(B.query(fallback_to_hash=True)[0][0], str)

    def test_label_to_uri(self):
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        g = onto.get_knowledge_graph(wf_dict)
        uri = asis.label_to_uri(g, "my_kinetic_energy_workflow")[0]
        label = str(g.value(uri, RDFS.label))
        self.assertEqual(label, "my_kinetic_energy_workflow")


if __name__ == "__main__":
    unittest.main()
