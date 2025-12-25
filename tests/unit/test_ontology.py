import unittest
from pathlib import Path
from textwrap import dedent

from pyshacl import validate
from rdflib import OWL, RDF, RDFS, BNode, Graph, Namespace
from rdflib.compare import graph_diff

from semantikon import ontology as onto
from semantikon.metadata import SemantikonURI, u
from semantikon.workflow import workflow

EX: Namespace = Namespace("http://example.org/")
PMD: Namespace = Namespace("https://w3id.org/pmd/co/PMD_")


prefixes = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix unit: <http://qudt.org/vocab/unit/> .
@prefix obi: <http://purl.obolibrary.org/obo/OBI_> .
@prefix sns: <http://semantikon.org/ontology/> .
@prefix ro: <http://purl.obolibrary.org/obo/RO_> .
@prefix pmd: <https://w3id.org/pmd/co/PMD_> .
@prefix ex: <http://example.org/> .
"""

sparql_prefixes = prefixes.replace("@prefix ", "PREFIX ").replace(" .\n", "\n")


def get_speed(
    distance: u(float, uri=PMD["0040001"], units="meter", label="Distance"),
    time: u(float, units="second"),
) -> u(float, units="meter/second"):
    """some random docstring"""
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
def wf_triples(a, b):
    a = f_triples(a, b)
    return a


class Meal:
    pass


def prepare_pizza() -> u(Meal, uri=EX.Pizza):
    return Meal()


def eat(meal: u(Meal, uri=EX.Meal)) -> str:
    return "I am full after eating "


@workflow
def eat_pizza():
    pizza = prepare_pizza()
    comment = eat(pizza)
    return comment


uri_color = SemantikonURI(EX.Color)
uri_cleaned = SemantikonURI(EX.Cleaned)


class Clothes:
    pass


def wash(
    clothes: Clothes,
) -> u(Clothes, triples=(EX.hasProperty, uri_cleaned), derived_from="inputs.clothes"):
    ...
    return clothes


def dye(clothes: Clothes, color="blue") -> u(
    Clothes,
    triples=(EX.hasProperty, uri_color),
    derived_from="inputs.clothes",
):
    ...
    return clothes


def sell(
    clothes: u(
        Clothes,
        restrictions=(
            ((OWL.onProperty, EX.hasProperty), (OWL.someValuesFrom, EX.Cleaned)),
            ((OWL.onProperty, EX.hasProperty), (OWL.someValuesFrom, EX.Color)),
        ),
    ),
) -> int:
    ...
    return 10


class TestOntology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.static_dir = Path(__file__).parent.parent / "static"

    def test_my_kinetic_energy_workflow_graph(self):
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        g = onto.get_knowledge_graph(wf_dict, include_t_box=False)

        with self.subTest("workflow instance exists"):
            workflows = list(g.subjects(RDF.type, onto.BASE.my_kinetic_energy_workflow))
            self.assertEqual(len(workflows), 1)

        wf = workflows[0]

        with self.subTest("workflow has both function executions as parts"):
            parts = list(g.objects(wf, onto.BFO["0000051"]))
            ke_calls = [
                p
                for p in parts
                if (
                    p,
                    RDF.type,
                    onto.BASE["my_kinetic_energy_workflow-get_kinetic_energy_0"],
                )
                in g
            ]
            speed_calls = [
                p
                for p in parts
                if (p, RDF.type, onto.BASE["my_kinetic_energy_workflow-get_speed_0"])
                in g
            ]
            self.assertEqual(len(ke_calls), 1)
            self.assertEqual(len(speed_calls), 1)

        ke_call = ke_calls[0]
        speed_call = speed_calls[0]

        with self.subTest("speed computation precedes kinetic energy computation"):
            self.assertIn(
                (speed_call, onto.BFO["0000063"], ke_call),
                g,
            )

        with self.subTest("functions are linked to python callables"):
            self.assertIn(
                (
                    ke_call,
                    onto.RO["0000057"],
                    onto.BASE[
                        f"{__name__}-get_kinetic_energy-not_defined".replace(".", "-")
                    ],
                ),
                g,
            )
            self.assertIn(
                (
                    speed_call,
                    onto.RO["0000057"],
                    onto.BASE[f"{__name__}-get_speed-not_defined".replace(".", "-")],
                ),
                g,
            )

        with self.subTest("kinetic energy output data has correct unit"):
            outputs = list(
                g.subjects(
                    RDF.type,
                    onto.BASE[
                        "my_kinetic_energy_workflow-get_kinetic_energy_0-outputs-output_data"
                    ],
                )
            )
            self.assertEqual(len(outputs), 1)
            self.assertIn(
                (
                    outputs[0],
                    Namespace("http://qudt.org/schema/qudt/").hasUnit,
                    Namespace("http://qudt.org/vocab/unit/").J,
                ),
                g,
            )
        g = onto.get_knowledge_graph(wf_dict)
        self.assertTrue(onto.validate_values(g)[0])

    def test_to_restrictions(self):
        # Common reference graph for single target class
        single_target_text = prefixes + dedent(
            """\
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
            text = prefixes + dedent(
                """\
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
            text = prefixes + dedent(
                """\
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
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        g = onto.get_knowledge_graph(wf_dict)
        shacl = onto.owl_restrictions_to_shacl(g)
        self.assertTrue(validate(g, shacl_graph=shacl)[0])

    def test_derives_from(self):
        wf_dict = wf_triples.serialize_workflow()
        g = onto.get_knowledge_graph(wf_dict, include_a_box=False)
        query = sparql_prefixes + dedent(
            """\
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
            onto.BASE["wf_triples-f_triples_0-outputs-a_data"],
        )
        g = onto.get_knowledge_graph(wf_dict, include_t_box=False)
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
        g = onto.get_knowledge_graph(wf_dict, include_t_box=False)

        with self.subTest("workflow instance exists"):
            workflows = list(g.subjects(RDF.type, onto.BASE.wf_triples))
            self.assertEqual(len(workflows), 1)

        wf = workflows[0]

        with self.subTest("workflow has a function activity part"):
            parts = list(g.objects(wf, onto.BFO["0000051"]))
            fn_activities = [
                p
                for p in parts
                if (
                    p,
                    onto.RO["0000057"],
                    onto.BASE[f"{__name__}-f_triples-not_defined".replace(".", "-")],
                )
                in g
            ]
            self.assertEqual(len(fn_activities), 1)

        with self.subTest("function output derives from input a"):
            outputs = list(
                g.subjects(RDF.type, onto.BASE["wf_triples-f_triples_0-outputs-a_data"])
            )
            self.assertEqual(len(outputs), 1)
            self.assertIn(
                (
                    outputs[0],
                    onto.RO["0001000"],
                    BNode(onto.BASE["wf_triples-inputs-a_data"]),
                ),
                g,
            )

        with self.subTest("cross-input data relations preserved"):
            self.assertIn(
                (
                    BNode(onto.BASE["wf_triples-inputs-b_data"]),
                    EX.relatedTo,
                    BNode(onto.BASE["wf_triples-inputs-a_data"]),
                ),
                g,
            )
            self.assertIn(
                (
                    outputs[0],
                    EX.hasSomeRelation,
                    BNode(onto.BASE["wf_triples-inputs-b_data"]),
                ),
                g,
            )
        g = onto.get_knowledge_graph(wf_dict)
        self.assertTrue(onto.validate_values(g)[0])

    def test_type_checking(self):
        wf_dict = eat_pizza.serialize_workflow()
        graph = onto.get_knowledge_graph(wf_dict)
        self.assertFalse(onto.validate_values(graph)[0])

    def test_restrictions(self):
        @workflow
        def my_correct_workflow(clothes: Clothes) -> int:
            dyed_clothes = dye(clothes)
            washed_clothes = wash(dyed_clothes)
            money = sell(washed_clothes)
            return money

        graph = onto.get_knowledge_graph(my_correct_workflow.serialize_workflow())
        self.assertTrue(onto.validate_values(graph)[0])

        @workflow
        def my_wrong_workflow(clothes: Clothes) -> int:
            washed_clothes = wash(clothes)
            money = sell(washed_clothes)
            return money

        graph = onto.get_knowledge_graph(my_wrong_workflow.serialize_workflow())
        self.assertFalse(onto.validate_values(graph)[0])


if __name__ == "__main__":
    unittest.main()
