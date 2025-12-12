import unittest
from pathlib import Path
from textwrap import dedent

from rdflib import OWL, RDFS, Graph, Namespace
from rdflib.compare import graph_diff

from semantikon import ontology as onto
from semantikon.metadata import u
from semantikon.workflow import workflow

EX: Namespace = Namespace("http://example.org/")
PMD: Namespace = Namespace("https://w3id.org/pmd/co/PMD_")


def get_speed(
    distance: u(float, uri=PMD["0040001"], units="meter"),
    time: u(float, units="second"),
) -> u(float, units="meter/second"):
    speed = distance / time
    return speed


def get_kinetic_energy(
    mass: u(float, uri=PMD["0020133"], units="kilogram"),
    velocity: u(float, units="meter/second"),
) -> u(float, uri=PMD["0020142"], units="joule"):
    return 0.5 * mass * velocity**2


@workflow
def my_kinetic_energy_workflow(distance: u(float, uri=PMD["0040001"]), time, mass):
    speed = get_speed(distance, time)
    kinetic_energy = get_kinetic_energy(mass, speed)
    return kinetic_energy


class TestOntology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.static_dir = Path(__file__).parent.parent / "static"

    def test_full_ontology(self):
        g_ref = Graph()
        with open(self.static_dir / "kinetic_energy_workflow.ttl", "r") as f:
            g_ref.parse(data=f.read(), format="turtle")
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        g = onto.get_knowledge_graph(wf_dict, t_box=True)
        _, in_first, in_second = graph_diff(g, g_ref)
        with self.subTest("Full ontology matches reference"):
            self.assertEqual(
                len(in_second), 0, msg=f"Missing triples: {in_second.serialize()}"
            )
        with self.subTest("Full ontology matches reference"):
            self.assertEqual(
                len(in_first), 0, msg=f"Unexpected triples: {in_first.serialize()}"
            )

    def test_to_restrictions(self):
        # Common reference graph for single target class
        single_target_text = dedent(
            """\
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        
        <http://example.org/origin> rdfs:subClassOf [ a owl:Restriction ;
                    owl:onProperty <http://example.org/some_predicate> ;
                    owl:someValuesFrom <http://example.org/destination> ],
                <http://example.org/my_class> .
        """
        )
        g_ref_single = Graph()
        g_ref_single.parse(data=single_target_text, format="turtle")

        with self.subTest("Single target class as list"):
            g = onto._to_owl_restriction(
                EX["some_predicate"], EX["destination"], base_node=EX["origin"]
            )
            g.add((EX["origin"], RDFS.subClassOf, EX["my_class"]))
            _, in_first, in_second = graph_diff(g, g_ref_single)
            self.assertEqual(
                len(in_second), 0, msg=f"Missing triples: {in_second.serialize()}"
            )
            self.assertEqual(
                len(in_first), 0, msg=f"Unexpected triples: {in_first.serialize()}"
            )

        with self.subTest("Multiple target classes"):
            text = dedent(
                """\
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            
            <http://example.org/origin> rdfs:subClassOf [ a owl:Restriction ;
                        owl:onProperty <http://example.org/some_predicate> ;
                        owl:someValuesFrom <http://example.org/dest1> ],
                    [ a owl:Restriction ;
                        owl:onProperty <http://example.org/some_predicate> ;
                        owl:someValuesFrom <http://example.org/dest2> ],
                    <http://example.org/my_class> .
            """
            )
            g_ref = Graph()
            g_ref.parse(data=text, format="turtle")
            g = Graph()
            for cl in [EX["dest1"], EX["dest2"]]:
                g += onto._to_owl_restriction(
                    EX["some_predicate"], cl, base_node=EX["origin"]
                )
            g.add((EX["origin"], RDFS.subClassOf, EX["my_class"]))
            _, in_first, in_second = graph_diff(g, g_ref)
            self.assertEqual(
                len(in_second), 0, msg=f"Missing triples: {in_second.serialize()}"
            )
            self.assertEqual(
                len(in_first), 0, msg=f"Unexpected triples: {in_first.serialize()}"
            )

        with self.subTest("owl:hasValue instead of owl:someValuesFrom"):
            text = dedent(
                """\
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            
            <http://example.org/origin> rdfs:subClassOf [ a owl:Restriction ;
                        owl:hasValue <http://example.org/destination> ;
                        owl:onProperty <http://example.org/some_predicate> ],
                    <http://example.org/my_class> .
            """
            )
            g_ref = Graph()
            g_ref.parse(data=text, format="turtle")
            g = onto._to_owl_restriction(
                EX["some_predicate"],
                EX["destination"],
                base_node=EX["origin"],
                restriction_type=OWL.hasValue,
            )
            g.add((EX["origin"], RDFS.subClassOf, EX["my_class"]))
            _, in_first, in_second = graph_diff(g, g_ref)
            self.assertEqual(
                len(in_second), 0, msg=f"Missing triples: {in_second.serialize()}"
            )
            self.assertEqual(
                len(in_first), 0, msg=f"Unexpected triples: {in_first.serialize()}"
            )

    def test_hash(self):
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        G = onto._wf_data_to_networkx(*onto.serialize_data(wf_dict))
        self.assertIsInstance(onto._get_graph_hash(G), str)
        self.assertEqual(len(onto._get_graph_hash(G)), 32)
        self.assertEqual(onto._get_graph_hash(G), "ca1e5a0ec85b1b83dc7061a9cc1f4113")
        self.assertIn(
            "dtype",
            G.nodes["my_kinetic_energy_workflow-get_speed_0-inputs-distance"],
            msg="dtype should not be deleted after hashing",
        )


if __name__ == "__main__":
    unittest.main()
