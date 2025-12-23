import unittest
from pathlib import Path
from textwrap import dedent

from pyshacl import validate
from rdflib import OWL, RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.compare import graph_diff

from semantikon import ontology as onto
from semantikon.metadata import u
from semantikon.workflow import workflow

EX: Namespace = Namespace("http://example.org/")
PMD: Namespace = Namespace("https://w3id.org/pmd/co/PMD_")


def get_speed(
    distance: u(float, uri=PMD["0040001"], units="meter", label="Distance"),
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


def f_triples(
    a: float, b: u(float, triples=("self", EX.relatedTo, "inputs.a"))
) -> u(float, triples=((EX.hasSomeRelation, "inputs.b")), derived_from="inputs.a"):
    return a


@workflow
def wf_triples(a):
    a = f_triples(a)
    return a


class TestOntology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.static_dir = Path(__file__).parent.parent / "static"

    def test_full_ontology(self):
        g_ref = Graph()
        with open(self.static_dir / "kinetic_energy_workflow.ttl", "r") as f:
            g_ref.parse(
                data=f.read().replace("__main__", __name__.replace(".", "-")),
                format="turtle",
            )
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
        g = onto.get_knowledge_graph(wf_dict, t_box=False)
        self.assertIn(
            (
                BNode(onto.BASE["my_kinetic_energy_workflow-inputs-distance_data"]),
                URIRef("http://qudt.org/vocab/unit/M"),
            ),
            list(g.subject_objects(onto.QUDT.hasUnit)),
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
                EX["origin"], EX["some_predicate"], EX["destination"]
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
                g += onto._to_owl_restriction(EX["origin"], EX["some_predicate"], cl)
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
                EX["origin"],
                EX["some_predicate"],
                EX["destination"],
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
        G = onto.serialize_and_convert_to_networkx(wf_dict)
        self.assertIsInstance(onto._get_graph_hash(G), str)
        self.assertEqual(len(onto._get_graph_hash(G)), 32)
        self.assertIn(
            "dtype",
            G.nodes["my_kinetic_energy_workflow-get_speed_0-inputs-distance"],
            msg="dtype should not be deleted after hashing",
        )
        self.assertEqual(
            onto._get_data_node(
                "my_kinetic_energy_workflow-get_kinetic_energy_0-inputs-velocity", G
            ),
            onto._get_data_node(
                "my_kinetic_energy_workflow-get_speed_0-outputs-speed", G
            ),
        )
        self.assertNotEqual(
            onto._get_data_node(
                "my_kinetic_energy_workflow-get_kinetic_energy_0-inputs-velocity", G
            ),
            onto._get_data_node(
                "my_kinetic_energy_workflow-get_kinetic_energy_0-outputs-output", G
            ),
        )

    def test_value(self):
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        G = onto.serialize_and_convert_to_networkx(wf_dict)
        wf_dict = my_kinetic_energy_workflow.run(1, 2, 3)
        G_run = onto.serialize_and_convert_to_networkx(wf_dict)
        self.assertEqual(
            onto._get_graph_hash(G),
            onto._get_graph_hash(G_run, with_global_inputs=False),
        )
        self.assertNotEqual(
            onto._get_graph_hash(G),
            onto._get_graph_hash(G_run, with_global_inputs=True),
        )

    def test_shacl_validation(self):
        Person = URIRef(EX + "Person")
        Email = URIRef(EX + "Email")
        hasEmail = URIRef(EX + "hasEmail")

        alice = URIRef(EX + "alice")
        bob = URIRef(EX + "bob")
        email1 = URIRef(EX + "email1")

        with self.subTest("Data conforms with valid property"):
            data = Graph()
            data.add((alice, RDF.type, Person))
            data.add((alice, hasEmail, email1))
            data.add((email1, RDF.type, Email))

            shapes = onto._to_shacl_shape(
                on_property=hasEmail,
                target_class=Email,
                base_node=Person,
            )

            conforms, _, _ = validate(
                data_graph=data,
                shacl_graph=shapes,
            )

            self.assertTrue(conforms)

        with self.subTest("Data fails when property is missing"):
            data = Graph()
            data.add((alice, RDF.type, Person))

            shapes = onto._to_shacl_shape(
                on_property=hasEmail,
                target_class=Email,
                base_node=Person,
            )

            conforms, _, report = validate(
                data_graph=data,
                shacl_graph=shapes,
            )

            self.assertFalse(conforms)
            self.assertIn("minCount", report)

        with self.subTest("Data fails when property value is wrong class"):
            data = Graph()
            data.add((alice, RDF.type, Person))
            data.add((alice, hasEmail, bob))  # bob not typed as Email

            shapes = onto._to_shacl_shape(
                on_property=hasEmail,
                target_class=Email,
                base_node=Person,
            )

            conforms, _, report = validate(
                data_graph=data,
                shacl_graph=shapes,
            )

            self.assertFalse(conforms)
            self.assertIn("class", report)

        with self.subTest("Non-target nodes are ignored"):
            data = Graph()
            data.add((bob, hasEmail, email1))
            data.add((email1, RDF.type, Email))

            shapes = onto._to_shacl_shape(
                on_property=hasEmail,
                target_class=Email,
                base_node=Person,
            )

            conforms, _, _ = validate(
                data_graph=data,
                shacl_graph=shapes,
            )

            self.assertTrue(conforms)

    def test_derives_from(self):
        wf_dict = wf_triples.serialize_workflow()
        g = onto.get_knowledge_graph(wf_dict, t_box=True)
        query = dedent(
            """\
        PREFIX ro: <http://purl.obolibrary.org/obo/RO_>
        SELECT ?main_class WHERE {
            ?derivedFrom owl:someValuesFrom ?input_class .
            ?derivedFrom owl:onProperty ro:0001000 .
            ?main_class rdfs:subClassOf ?derivedFrom .
            ?input_class rdfs:subClassOf obi:0001933 .
        }
        """
        )
        self.assertEqual(len(g.query(query)), 1)
        self.assertEqual(
            list(g.query(query))[0]["main_class"],
            BNode(onto.BASE["wf_triples-f_triples_0-outputs-a_data"]),
        )
        g = onto.get_knowledge_graph(wf_dict, t_box=False)
        result = list(
            g.predicates(
                BNode(onto.BASE["wf_triples-f_triples_0-outputs-a_data"]),
                BNode(onto.BASE["wf_triples-inputs-a_data"]),
            )
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], onto.SNS.derives_from)

    def test_triples(self):
        wf_dict = wf_triples.serialize_workflow()
        g = onto.get_knowledge_graph(wf_dict, t_box=True)
        query = dedent(
            """\
        PREFIX ro: <http://purl.obolibrary.org/obo/RO_>
        PREFIX ex: <http://example.org/>
        PREFIX obi: <http://purl.obolibrary.org/obo/OBI_>
        SELECT ?input_label ?output_label WHERE {
            ?input_data_class rdfs:label ?input_label .
            ?input_data_class rdfs:subClassOf ?input_data_node .
            ?input_data_node owl:onProperty obi:0000293 .
            ?input_data_node owl:someValuesFrom ?input_b .
            ?has_some_relation owl:someValuesFrom ?input_b .
            ?has_some_relation owl:onProperty ex:hasSomeRelation .
            ?data_class_node rdfs:subClassOf ?has_some_relation .
            ?data_class owl:someValuesFrom ?data_class_node .
            ?data_class owl:onProperty obi:0000299 .
            ?output_class rdfs:subClassOf ?data_class .
            ?output_class rdfs:label ?output_label
        }
        """
        )
        self.assertEqual(list(g.query(query)), [(Literal("b"), Literal("a"))])
        g = onto.get_knowledge_graph(wf_dict, t_box=False)
        self.assertEqual(
            list(g.subject_objects(EX.relatedTo)),
            [
                (
                    BNode(onto.BASE["wf_triples-f_triples_0-inputs-b_data"]),
                    BNode(onto.BASE["wf_triples-inputs-a_data"]),
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
