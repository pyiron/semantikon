import unittest
from pathlib import Path
from textwrap import dedent
from typing import Annotated

from pyshacl import validate
from rdflib import OWL, RDF, RDFS, Graph, Literal, Namespace
from rdflib.compare import graph_diff

from semantikon import ontology as onto
from semantikon.metadata import SemantikonURI, meta
from semantikon.visualize import visualize
from semantikon.workflow import workflow

EX: Namespace = Namespace("http://example.org/")
PMD: Namespace = Namespace("https://w3id.org/pmd/co/PMD_")


prefixes = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix unit: <http://qudt.org/vocab/unit/> .
@prefix obi: <http://purl.obolibrary.org/obo/OBI_> .
@prefix sns: <http://pyiron.org/ontology/> .
@prefix ro: <http://purl.obolibrary.org/obo/RO_> .
@prefix pmd: <https://w3id.org/pmd/co/PMD_> .
@prefix ex: <http://example.org/> .
"""

sparql_prefixes = prefixes.replace("@prefix ", "PREFIX ").replace(" .\n", "\n")


def get_speed(
    distance: Annotated[
        float, {"uri": PMD["0040001"], "units": "meter", "label": "Distance"}
    ],
    time: Annotated[float, {"units": "second"}],
) -> Annotated[float, {"units": "meter/second", "uri": EX.Velocity}]:
    """some random docstring"""
    speed = distance / time
    return speed


@meta(uri=EX.get_kinetic_energy)
def get_kinetic_energy(
    mass: Annotated[float, {"uri": PMD["0020133"], "units": "kilogram"}],
    velocity: Annotated[float, {"units": "meter/second", "uri": EX.Velocity}],
) -> Annotated[float, {"uri": PMD["0020142"], "units": "joule"}]:
    return 0.5 * mass * velocity**2


@workflow
def my_kinetic_energy_workflow(
    distance: Annotated[float, {"uri": PMD["0040001"]}], time, mass
):
    speed = get_speed(distance, time)
    kinetic_energy = get_kinetic_energy(mass, speed)
    return kinetic_energy


def get_kinetic_energy_wrong_units(
    mass: Annotated[float, {"uri": PMD["0020133"], "units": "kilogram"}],
    velocity: Annotated[float, {"units": "angstrom", "uri": EX.Velocity}],
) -> Annotated[float, {"uri": PMD["0020142"], "units": "joule"}]:
    return 0.5 * mass * velocity**2


def get_kinetic_energy_wrong_uri(
    mass: Annotated[float, {"units": "kilogram"}],
    velocity: Annotated[float, {"units": "meter/second", "uri": EX.WrongURI}],
) -> Annotated[float, {"uri": PMD["0020142"], "units": "joule"}]:
    return 0.5 * mass * velocity**2


def get_speed_not_annotated(distance, time) -> float:
    return distance / time


def f_triples(
    a: float, b: Annotated[float, {"triples": ("self", EX.relatedTo, "inputs.a")}]
) -> Annotated[
    float, {"triples": ((EX.hasSomeRelation, "inputs.b")), "derived_from": "inputs.a"}
]:
    return a


@workflow
def wf_triples(a, b):
    a = f_triples(a, b)
    return a


class Meal:
    pass


def prepare_pizza() -> Annotated[Meal, {"uri": EX.Pizza}]:
    return Meal()


def eat(meal: Annotated[Meal, {"uri": EX.Meal}]) -> str:
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
) -> Annotated[
    Clothes,
    {"triples": (EX.hasProperty, uri_cleaned), "derived_from": "inputs.clothes"},
]:
    ...
    return clothes


def dye(
    clothes: Clothes, color="blue"
) -> Annotated[
    Clothes, {"triples": (EX.hasProperty, uri_color), "derived_from": "inputs.clothes"}
]:
    ...
    return clothes


def sell(
    clothes: Annotated[
        Clothes,
        {
            "restrictions": (
                ((OWL.onProperty, EX.hasProperty), (OWL.someValuesFrom, EX.Cleaned)),
                ((OWL.onProperty, EX.hasProperty), (OWL.someValuesFrom, EX.Color)),
            )
        },
    ],
) -> int:
    ...
    return 10


def sell_without_color(
    clothes: Annotated[
        Clothes,
        # Different shape from above
        {
            "restrictions": (
                (OWL.onProperty, EX.hasProperty),
                (OWL.someValuesFrom, EX.Cleaned),
            )
        },
    ],
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

        @workflow
        def workflow_with_default_values(distance=2, time=1, mass=4):
            speed = get_speed(distance, time)
            kinetic_energy = get_kinetic_energy(mass, speed)
            return kinetic_energy

        wf_dict = workflow_with_default_values.serialize_workflow()
        wf_dict_run = workflow_with_default_values.run(distance=2, time=1, mass=4)
        G = onto.serialize_and_convert_to_networkx(wf_dict)
        G_run = onto.serialize_and_convert_to_networkx(wf_dict_run)
        self.assertEqual(onto._get_graph_hash(G), onto._get_graph_hash(G_run))

    def test_hash_with_value(self):
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
        query = sparql_prefixes + dedent(
            """
        ASK WHERE {
            ?output a sns:wf_triples-f_triples_0-outputs-a_data .
            ?input a sns:wf_triples-inputs-a_data .
            ?output ro:0001000 ?input .
        }
        """
        )
        self.assertTrue(g.query(query).askAnswer)

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

        with self.subTest("cross-input data relations preserved"):
            query = sparql_prefixes + dedent(
                """
            ASK WHERE {
                ?b_data a sns:wf_triples-inputs-b_data .
                ?a_data a sns:wf_triples-inputs-a_data .
                ?b_data ex:relatedTo ?a_data .
            }
            """
            )
            self.assertTrue(g.query(query).askAnswer)
            query = sparql_prefixes + dedent(
                """
            ASK WHERE {
                ?output a sns:wf_triples-f_triples_0-outputs-a_data .
                ?input a sns:wf_triples-inputs-b_data .
                ?output ex:hasSomeRelation ?input .
            }
            """
            )
            self.assertTrue(g.query(query).askAnswer)
        g = onto.get_knowledge_graph(wf_dict)
        self.assertTrue(onto.validate_values(g)[0])

    def test_type_checking(self):
        wf_dict = eat_pizza.serialize_workflow()
        graph = onto.get_knowledge_graph(wf_dict)
        self.assertFalse(onto.validate_values(graph)[0])
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        graph = onto.get_knowledge_graph(wf_dict)
        verdict, _, report = onto.validate_values(graph)
        self.assertTrue(verdict, msg=report)

        @workflow
        def my_kinetic_energy_workflow_wrong_units(
            distance: Annotated[float, {"uri": PMD["0040001"]}], time, mass
        ):
            speed = get_speed(distance, time)
            kinetic_energy = get_kinetic_energy_wrong_units(mass, speed)
            return kinetic_energy

        wf_dict = my_kinetic_energy_workflow_wrong_units.serialize_workflow()
        graph = onto.get_knowledge_graph(wf_dict)
        self.assertFalse(onto.validate_values(graph)[0])

        @workflow
        def my_kinetic_energy_workflow_wrong_uri(
            distance: Annotated[float, {"uri": PMD["0040001"]}], time, mass
        ):
            speed = get_speed(distance, time)
            kinetic_energy = get_kinetic_energy_wrong_uri(mass, speed)
            return kinetic_energy

        wf_dict = my_kinetic_energy_workflow_wrong_uri.serialize_workflow()
        graph = onto.get_knowledge_graph(wf_dict)
        self.assertFalse(onto.validate_values(graph)[0])

        @workflow
        def my_kinetic_energy_workflow_not_annotated(
            distance: Annotated[float, {"uri": PMD["0040001"]}], time, mass
        ):
            speed = get_speed_not_annotated(distance, time)
            kinetic_energy = get_kinetic_energy(mass, speed)
            return kinetic_energy

        wf_dict = my_kinetic_energy_workflow_not_annotated.serialize_workflow()
        graph = onto.get_knowledge_graph(wf_dict)
        self.assertFalse(onto.validate_values(graph, strict_typing=True)[0])
        self.assertTrue(onto.validate_values(graph, strict_typing=False)[0])

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

        @workflow
        def my_simple_workflow(clothes: Clothes) -> int:
            washed_clothes = wash(clothes)
            money = sell_without_color(washed_clothes)
            return money

        graph = onto.get_knowledge_graph(my_simple_workflow.serialize_workflow())
        self.assertTrue(onto.validate_values(graph)[0])

    def test_visualize(self):
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        g = onto.get_knowledge_graph(wf_dict)
        from graphviz.graphs import Digraph

        self.assertIsInstance(visualize(g), Digraph)

    def test_docstring(self):
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        g = onto.get_knowledge_graph(wf_dict)
        bnode = list(g.subjects(RDF.type, onto.SNS.textual_entity))
        self.assertEqual(len(bnode), 1)
        self.assertEqual(g.value(bnode[0], RDF.value), Literal("some random docstring"))

    def test_function_metadata(self):
        wf_dict = my_kinetic_energy_workflow.serialize_workflow()
        graph = onto.get_knowledge_graph(wf_dict)
        # Check that the main subject exists
        main_subject = onto.BASE[
            f"{__name__}-get_kinetic_energy-not_defined".replace(".", "-")
        ]
        self.assertIn((main_subject, RDF.type, onto.IAO["0000591"]), graph)
        self.assertIn((main_subject, RDFS.label, Literal("get_kinetic_energy")), graph)
        self.assertIn((main_subject, onto.IAO["0000136"], EX.get_kinetic_energy), graph)

        # Check input specifications
        input_specifications = list(
            graph.objects(main_subject, onto.BASE.has_parameter_specification)
        )
        self.assertEqual(len(input_specifications), 3)  # 2 inputs and 1 output

        # Check the first input specification (mass)
        mass_spec = list(graph.subjects(onto.IAO["0000136"], onto.PMD["0020133"]))
        self.assertEqual(len(mass_spec), 1)
        self.assertIn((mass_spec[0], RDF.type, onto.BASE.input_specification), graph)
        self.assertIn((mass_spec[0], RDFS.label, Literal("mass")), graph)
        self.assertIn(
            (mass_spec[0], onto.BASE.has_parameter_position, Literal(0)), graph
        )

        # Check the output specification
        output_spec = list(graph.subjects(onto.IAO["0000136"], onto.PMD["0020142"]))
        self.assertEqual(len(output_spec), 1)
        self.assertIn((output_spec[0], RDF.type, onto.BASE.output_specification), graph)
        self.assertIn((output_spec[0], RDFS.label, Literal("output")), graph)
        self.assertIn(
            (output_spec[0], onto.BASE.has_parameter_position, Literal(0)), graph
        )

    def test_run(self):
        g_run = onto.get_knowledge_graph(my_kinetic_energy_workflow.run(2, 1, 4))
        query = (
            sparql_prefixes
            + """
        SELECT ?node ?value WHERE {
          ?bnode a ?node ;
            rdf:value ?value .
        }
        """
        )
        results = list(g_run.query(query))
        for tag, value in zip(
            [
                "my_kinetic_energy_workflow-inputs-mass_data",
                "my_kinetic_energy_workflow-get_kinetic_energy_0-outputs-output_data",
                "my_kinetic_energy_workflow-inputs-time_data",
                "my_kinetic_energy_workflow-get_speed_0-outputs-speed_data",
                "my_kinetic_energy_workflow-inputs-distance_data",
            ],
            [4, 8.0, 1, 2.0, 2],
        ):
            with self.subTest(f"Checking value for {tag}"):
                matched = [
                    (str(row["node"]), row["value"])
                    for row in results
                    if str(row["node"]).endswith(tag)
                ]
                self.assertEqual(len(matched), 1)
                self.assertEqual(matched[0][1].toPython(), value)


if __name__ == "__main__":
    unittest.main()
