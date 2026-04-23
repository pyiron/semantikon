import copy
import os
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Annotated

from pyshacl import validate
from rdflib import OWL, RDF, RDFS, SH, BNode, Graph, Literal, Namespace
from rdflib.compare import graph_diff

from semantikon import ontology as onto
from semantikon.metadata import SemantikonURI, meta
from semantikon.visualize import visualize_recipe
from semantikon.workflow import workflow

EX: Namespace = Namespace("http://example.org/")
PMD: Namespace = Namespace("https://w3id.org/pmd/co/PMD_")
UNIT: Namespace = Namespace("http://qudt.org/vocab/unit/")


prefixes = """
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix unit: <http://qudt.org/vocab/unit/> .
@prefix obi: <http://purl.obolibrary.org/obo/OBI_> .
@prefix sns: <http://pyiron.org/ontology/> .
@prefix ro: <http://purl.obolibrary.org/obo/RO_> .
@prefix pmdco: <https://w3id.org/pmd/co/PMD_> .
@prefix ex: <http://example.org/> .
"""

sparql_prefixes = prefixes.replace("@prefix ", "PREFIX ").replace(" .\n", "\n")


def add_one(x):
    y = x + 1
    return y


@workflow
def add_two(a):
    b = add_one(a)
    c = add_one(b)
    return c


@workflow
def add_three(alpha):
    beta = add_two(alpha)
    gamma = add_two(beta)
    return gamma


def upstream_node(a) -> Annotated[object, {"uri": EX.Upstream}]:
    b = a
    return b


def downstream_node(x: Annotated[object, {"uri": EX.Downstream}]):
    y = x
    return y


@workflow
def undefined_specificity(a):
    b = upstream_node(a)
    y = downstream_node(b)
    return y


@workflow
def input_uri_nonalignment(
        a: Annotated[object, {"uri": EX.SomethingOtherThanDownstream}]
):
    y = downstream_node(a)
    return y


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


@dataclass
class NewSpeedData:
    distance: Annotated[float, {"uri": PMD["0040001"], "units": "meter"}]
    time: float = field(metadata={"units": "second"})


def get_speed_with_dataclass(speed_data: NewSpeedData):
    speed = speed_data.distance / speed_data.time
    return speed


def f_triples(
    a: Annotated[float, {"uri": EX.Something}],
    b: Annotated[float, {"triples": ("self", EX.relatedTo, "inputs.a")}],
) -> Annotated[
    float,
    {
        "triples": ((EX.hasSomeRelation, "inputs.b")),
        "derived_from": "inputs.a",
        "uri": EX.SomethingElse,
    },
]:
    return a


@workflow
def wf_triples(a, b):
    a = f_triples(a, b)
    return a


@workflow
def wf_nested_triples(a, b):
    result = wf_triples(a, b)
    return result


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


class Drink:
    pass


@workflow
def swallow_pizza_painfully(drink: Annotated[Drink, {"uri": EX.Water}]):
    x = eat(drink)
    return x


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


def sell_with_shacl(
    clothes: Annotated[
        Clothes,
        {
            "restrictions": (
                (
                    (SH.path, EX.hasProperty),
                    (SH.minCount, Literal(1)),
                    (SH["class"], EX.Cleaned),
                ),
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


@dataclass
class Input:
    T: Annotated[float, {"units": "kelvin"}]
    n: int

    @dataclass
    class parameters:
        a: int = 2

    class not_dataclass:
        b: int = 3


@dataclass
class Output:
    E: Annotated[float, {"units": "electron_volt", "uri": EX.Energy}]
    L: Annotated[float, {"units": "nanometer"}]


def run_md(inp: Input, E=1.0) -> Output:
    out = Output(E=E, L=2.0)
    return out


class Unhashable:
    pass


def get_unhashable(uh: Unhashable):
    return uh


def add(a: int, b: int) -> int:
    return a + b


def mul(x: int, y: int) -> int:
    return x * y


@workflow
def multiple_connection(a, b):
    c = add(a, b)
    d = mul(a, c)
    e = add(d, c)
    return e


@workflow
def get_run_md(inp: Input, E=1.0):
    result = run_md(inp)
    return result


@workflow
def workflow_with_default_values(distance=2, time=1, mass=4):
    speed = get_speed(distance, time)
    kinetic_energy = get_kinetic_energy(mass, speed)
    return kinetic_energy


@workflow
def my_correct_workflow(clothes: Clothes) -> int:
    dyed_clothes = dye(clothes)
    washed_clothes = wash(dyed_clothes)
    money = sell(washed_clothes)
    return money


@workflow
def my_wrong_workflow(clothes: Clothes) -> int:
    washed_clothes = wash(clothes)
    money = sell(washed_clothes)
    return money


@workflow
def my_simple_workflow(clothes: Clothes) -> int:
    washed_clothes = wash(clothes)
    money = sell_without_color(washed_clothes)
    return money


@workflow
def my_shacl_workflow(clothes: Clothes) -> int:
    washed_clothes = wash(clothes)
    money = sell_with_shacl(washed_clothes)
    return money


@workflow
def my_shacl_wrong_workflow(clothes: Clothes) -> int:
    money = sell_with_shacl(clothes)
    return money


@workflow
def my_kinetic_energy_workflow_wrong_units(
    distance: Annotated[float, {"uri": PMD["0040001"]}], time, mass
):
    speed = get_speed(distance, time)
    kinetic_energy = get_kinetic_energy_wrong_units(mass, speed)
    return kinetic_energy


@workflow
def my_kinetic_energy_workflow_wrong_uri(
    distance: Annotated[float, {"uri": PMD["0040001"]}], time, mass
):
    speed = get_speed(distance, time)
    kinetic_energy = get_kinetic_energy_wrong_uri(mass, speed)
    return kinetic_energy


@workflow
def my_kinetic_energy_workflow_not_annotated(
    distance: Annotated[float, {"uri": PMD["0040001"]}], time, mass
):
    speed = get_speed_not_annotated(distance, time)
    kinetic_energy = get_kinetic_energy(mass, speed)
    return kinetic_energy


@workflow
def workflow_with_dataclass(speed_data, mass):
    speed = get_speed_with_dataclass(speed_data)
    kinetic_energy = get_kinetic_energy(mass, speed)
    return kinetic_energy


@workflow
def my_unhashable_inputs(uh):
    result = get_unhashable(uh)
    return result


class TestOntology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        cls.static_dir = Path(__file__).parent.parent / "static"

    def test_nested_topology(self):
        wf_dict = add_three.get_semantikon_dict()

        def _label(fnc) -> str:
            return fnc.__name__ + "_0"

        with self.subTest("Basic construction"):
            g = onto.get_knowledge_graph(wf_dict=wf_dict)
            validation = onto.validate_values(g)
            self.assertTrue(
                validation[0],
                msg="Trivial deeply nested topology should parse and validates",
            )

        subgraph_label = add_two.__name__ + "_0"
        subgraph_edges = wf_dict["nodes"][subgraph_label]["edges"]

        with self.subTest("Too many inputs"):
            extra_input = ("inputs.a", "add_one_1.inputs.x")
            overloaded_input_edges = list(subgraph_edges)
            overloaded_input_edges.append(extra_input)
            overloaded_dict = copy.deepcopy(wf_dict)
            overloaded_dict["nodes"][subgraph_label]["edges"] = overloaded_input_edges

            with self.assertRaises(
                AssertionError,
                msg="Providing an input with two legitimate sources (a peer source and "
                "a parent source, in this case) should force an error. It would be "
                "nice to hit the ValueError in __, but we hit an assertion in "
                "`_get_data_node` first.",
            ):
                onto.get_knowledge_graph(wf_dict=overloaded_dict)

    def test_uri_specificity(self):
        for source_more_specific in [True, False]:
            with self.subTest(f"source_more_specific={source_more_specific}"):
                d = undefined_specificity.get_semantikon_dict()
                g = onto.get_knowledge_graph(d)

                if source_more_specific:
                    g.add((EX.Upstream, RDFS.subClassOf, EX.Downstream))
                else:
                    g.add((EX.Downstream, RDFS.subClassOf, EX.Upstream))
                v = onto.validate_values(g)
                self.assertEqual(
                    v[0],
                    source_more_specific,
                    msg="Dataflow type validation requires that upstream sources be "
                    "as-or-more-specific than downstream targets. This requirement "
                    "also holds for 'meaning' types carried by the `uri=` annotation. "
                    "Here, upstream and downstream URIs are not the same, and thus we "
                    "require specificity directionality to hold.",
                )

    def test_enforce_subgraph_uri_alignment_toggle(self):
        d = input_uri_nonalignment.get_semantikon_dict()

        with self.subTest("Misaligned and enforced"):
            g = onto.get_knowledge_graph(d, enforce_subgraph_uri_alignment=True)
            valid, _, report = onto.validate_values(g)
            self.assertFalse(valid)
            self.assertIn("input_uri_nonalignment-inputs-a_data", report)

        with self.subTest("Misaligned and not enforced"):
            g = onto.get_knowledge_graph(d, enforce_subgraph_uri_alignment=False)
            valid, _, _ = onto.validate_values(g)
            self.assertTrue(valid)

    def test_enforce_subgraph_uri_alignment_subclassing(self):
        d = input_uri_nonalignment.get_semantikon_dict()
        g = onto.get_knowledge_graph(d, enforce_subgraph_uri_alignment=True)
        g.add((EX.SomethingOtherThanDownstream, RDFS.subClassOf, EX.Downstream))
        valid, _, _ = onto.validate_values(g)
        self.assertTrue(valid)

    def test_my_kinetic_energy_workflow_graph(self):
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        g = onto.get_knowledge_graph(wf_dict)

        query = f"""
        PREFIX qudt: <http://qudt.org/schema/qudt/>
        PREFIX unit: <http://qudt.org/vocab/unit/>
        PREFIX sns: <{onto.BASE}>
        PREFIX obi: <http://purl.obolibrary.org/obo/OBI_>

        ASK {{
            ?output a sns:Wc59275b7_my_kinetic_energy_workflow-outputs-kinetic_energy .
            ?output ro:0000057 ?data .
            ?data qudt:hasUnit unit:J .
        }}"""
        self.assertTrue(
            g.query(query).askAnswer,
            msg=g.serialize() + """
            The reason why this test failed is probably because the representation
            of the workflow graph changed and therefore also its hash value
            (starting with W in the URI)
            """,
        )
        self.assertTrue(onto.validate_values(g)[0])

    def test_to_restrictions(self):
        # Common reference graph for single target class
        single_target_text = prefixes + dedent("""\
        <http://example.org/origin> rdfs:subClassOf [ a owl:Restriction ;
                    owl:onProperty <http://example.org/some_predicate> ;
                    owl:someValuesFrom <http://example.org/destination> ],
                <http://example.org/my_class> .
        """)
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
            text = prefixes + dedent("""\
            <http://example.org/origin> rdfs:subClassOf [ a owl:Restriction ;
                        owl:onProperty <http://example.org/some_predicate> ;
                        owl:someValuesFrom <http://example.org/dest1> ],
                    [ a owl:Restriction ;
                        owl:onProperty <http://example.org/some_predicate> ;
                        owl:someValuesFrom <http://example.org/dest2> ],
                    <http://example.org/my_class> .
            """)
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
            text = prefixes + dedent("""\
            <http://example.org/origin> rdfs:subClassOf [ a owl:Restriction ;
                        owl:hasValue <http://example.org/destination> ;
                        owl:onProperty <http://example.org/some_predicate> ],
                    <http://example.org/my_class> .
            """)
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
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        G = onto.serialize_and_convert_to_networkx(wf_dict, hash_data=False)
        self.assertIsInstance(onto._get_graph_hash(G), str)
        self.assertEqual(len(onto._get_graph_hash(G)), 32)
        self.assertIn(
            "dtype",
            G.nodes["my_kinetic_energy_workflow-get_speed_0-inputs-distance"],
            msg="dtype should not be deleted after hashing",
        )
        self.assertEqual(
            G._get_data_node(
                "my_kinetic_energy_workflow-get_kinetic_energy_0-inputs-velocity"
            ),
            G._get_data_node("my_kinetic_energy_workflow-get_speed_0-outputs-speed"),
        )
        self.assertNotEqual(
            G._get_data_node(
                "my_kinetic_energy_workflow-get_kinetic_energy_0-inputs-velocity"
            ),
            G._get_data_node(
                "my_kinetic_energy_workflow-get_kinetic_energy_0-outputs-kinetic_energy"
            ),
        )
        wf_dict_one = my_kinetic_energy_workflow.run(distance=1.0, time=2.0, mass=3.0)
        wf_dict_two = my_kinetic_energy_workflow.run(distance=4.0, time=5.0, mass=6.0)
        G_one = onto.serialize_and_convert_to_networkx(wf_dict_one, hash_data=True)
        G_two = onto.serialize_and_convert_to_networkx(wf_dict_two, hash_data=True)
        self.assertEqual(
            onto._get_graph_hash(G_one, with_global_inputs=False),
            onto._get_graph_hash(G_two, with_global_inputs=False),
        )

        wf_dict = workflow_with_default_values.get_semantikon_dict()
        wf_dict_run = workflow_with_default_values.run(distance=2, time=1, mass=4)
        G = onto.serialize_and_convert_to_networkx(wf_dict, hash_data=False)
        G_run = onto.serialize_and_convert_to_networkx(wf_dict_run, hash_data=False)
        self.assertEqual(onto._get_graph_hash(G), onto._get_graph_hash(G_run))
        G_hash = onto.serialize_and_convert_to_networkx(wf_dict_run, hash_data=True)
        self.assertDictEqual(
            {key.split("@")[1]: value for key, value in G_hash.get_hash_dict().items()},
            {"kinetic_energy": 8.0, "speed": 2.0},
        )
        with self.assertRaises(TypeError):
            wf_dict["inputs"]["distance"]["default"] = NewSpeedData
            G = onto.serialize_and_convert_to_networkx(wf_dict, hash_data=True)
            onto._get_graph_hash(G, with_global_inputs=True)
        with self.assertRaises(TypeError):
            wf_dict["inputs"]["distance"]["default"] = BNode()
            G = onto.serialize_and_convert_to_networkx(wf_dict, hash_data=True)
            onto._get_graph_hash(G, with_global_inputs=True)

    def test_hash_with_value(self):
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        G = onto.serialize_and_convert_to_networkx(wf_dict, hash_data=False)
        wf_dict = my_kinetic_energy_workflow.run(distance=1, time=2, mass=3)
        G_run = onto.serialize_and_convert_to_networkx(wf_dict, hash_data=False)
        self.assertEqual(
            onto._get_graph_hash(G, with_global_inputs=False),
            onto._get_graph_hash(G_run, with_global_inputs=False),
        )
        self.assertNotEqual(
            onto._get_graph_hash(G_run, with_global_inputs=False),
            onto._get_graph_hash(G_run, with_global_inputs=True),
        )

    def test_shacl_validation(self):
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        g = onto.get_knowledge_graph(wf_dict)
        shacl = onto.owl_restrictions_to_shacl(g)
        self.assertTrue(validate(g, shacl_graph=shacl)[0])

    def test_derives_from(self):
        wf_dict = wf_triples.get_semantikon_dict()
        g = onto.get_knowledge_graph(wf_dict, include_a_box=False, prefix="T")
        query = sparql_prefixes + dedent("""\
        SELECT ?main_class WHERE {
            ?derivedFrom owl:someValuesFrom ?input_class .
            ?derivedFrom owl:onProperty ro:0001000 .
            ?main_class rdfs:subClassOf ?derivedFrom .
            ?input_class rdfs:subClassOf obi:0001933 .
        }
        """)
        self.assertEqual(len(g.query(query)), 1)
        self.assertEqual(
            list(g.query(query))[0]["main_class"],
            onto.BASE["T_wf_triples-f_triples_0-outputs-a_data"],
        )
        g = onto.get_knowledge_graph(wf_dict, include_t_box=False, prefix="T")
        query = sparql_prefixes + dedent("""
        ASK WHERE {
            ?output a sns:T_wf_triples-f_triples_0-outputs-a_data .
            ?input a sns:T_wf_triples-inputs-a_data .
            ?output ro:0001000 ?input .
        }
        """)
        self.assertTrue(g.query(query).askAnswer)

    def test_triples(self):
        wf_dict = wf_triples.get_semantikon_dict()
        g = onto.get_knowledge_graph(wf_dict, include_t_box=False, prefix="T")

        with self.subTest("workflow instance exists"):
            workflows = list(g.subjects(RDF.type, onto.BASE.T_wf_triples))
            self.assertEqual(len(workflows), 1)

        wf = workflows[0]

        with self.subTest("workflow has a function activity part"):
            parts = list(g.objects(wf, onto.BFO["0000051"]))
            fn_activities = [
                p
                for p in parts
                if (
                    p,
                    onto.RO["0000059"],
                    onto.BASE[f"{__name__}-f_triples-not_defined".replace(".", "-")],
                )
                in g
            ]
            self.assertEqual(len(fn_activities), 1)

        with self.subTest("cross-input data relations preserved"):
            query = sparql_prefixes + dedent("""
            ASK {
                ?b_data a sns:T_wf_triples-inputs-b_data .
                ?a_data a sns:T_wf_triples-inputs-a_data .
                ?b_data ex:relatedTo ?a_data .
            }
            """)
            self.assertTrue(g.query(query).askAnswer)
            query = sparql_prefixes + dedent("""
            ASK {
                ?output a sns:T_wf_triples-f_triples_0-outputs-a_data .
                ?input a sns:T_wf_triples-inputs-b_data .
                ?output ex:hasSomeRelation ?input .
            }
            """)
            self.assertTrue(g.query(query).askAnswer)
        g = onto.get_knowledge_graph(wf_dict)
        self.assertTrue(onto.validate_values(g)[0])

    def test_type_checking(self):
        wf_dict = eat_pizza.get_semantikon_dict()
        graph = onto.get_knowledge_graph(wf_dict)
        self.assertFalse(onto.validate_values(graph)[0], msg=graph.serialize())
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        graph = onto.get_knowledge_graph(wf_dict)
        verdict, _, report = onto.validate_values(graph)
        self.assertTrue(verdict, msg=report)

        wf_dict = my_kinetic_energy_workflow_wrong_units.get_semantikon_dict()
        graph = onto.get_knowledge_graph(wf_dict)
        self.assertFalse(onto.validate_values(graph)[0])

        wf_dict = my_kinetic_energy_workflow_wrong_uri.get_semantikon_dict()
        graph = onto.get_knowledge_graph(wf_dict)
        self.assertFalse(onto.validate_values(graph)[0])

        wf_dict = my_kinetic_energy_workflow_not_annotated.get_semantikon_dict()
        graph = onto.get_knowledge_graph(wf_dict)
        self.assertFalse(onto.validate_values(graph, strict_typing=True)[0])
        self.assertTrue(onto.validate_values(graph, strict_typing=False)[0])

    def test_restrictions(self):
        graph = onto.get_knowledge_graph(my_correct_workflow.get_semantikon_dict())
        self.assertTrue(onto.validate_values(graph)[0])

        graph = onto.get_knowledge_graph(my_wrong_workflow.get_semantikon_dict())
        self.assertFalse(onto.validate_values(graph)[0])

        graph = onto.get_knowledge_graph(my_simple_workflow.get_semantikon_dict())
        self.assertTrue(onto.validate_values(graph)[0])

        graph = onto.get_knowledge_graph(my_shacl_workflow.get_semantikon_dict())
        verdict, _, report = onto.validate_values(graph)
        self.assertTrue(verdict, msg=report)

        graph = onto.get_knowledge_graph(my_shacl_wrong_workflow.get_semantikon_dict())
        self.assertFalse(onto.validate_values(graph)[0])

    def test_visualize(self):
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        g = onto.get_knowledge_graph(wf_dict)
        from graphviz.graphs import Digraph

        self.assertIsInstance(visualize_recipe(g), Digraph)

    def test_docstring(self):
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        g = onto.get_knowledge_graph(wf_dict)
        bnode = list(g.subjects(RDF.type, onto.SNS.textual_entity))
        self.assertEqual(len(bnode), 1)
        self.assertEqual(
            g.value(bnode[0], onto.SNS.has_value), Literal("some random docstring")
        )

    def test_function_metadata(self):
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        graph = onto.get_knowledge_graph(wf_dict)
        # Check that the main subject exists
        main_subject = onto.BASE[
            f"{__name__}-get_kinetic_energy-not_defined".replace(".", "-")
        ]
        self.assertIn((main_subject, RDF.type, onto.PMD["0000010"]), graph)

        # Check input specifications
        input_specifications = list(graph.objects(main_subject, onto.SNS.has_part))
        self.assertEqual(len(input_specifications), 3)  # 2 inputs and 1 output

        # Check the first input specification (mass)
        query = sparql_prefixes + """
            SELECT ?input WHERE {
              ?input a ?bnode .
              ?bnode a owl:Restriction .
              ?bnode owl:onProperty iao:0000136 .
              ?bnode owl:allValuesFrom pmdco:0020133 .
            }"""
        mass_spec = list(graph.query(query))
        self.assertEqual(len(mass_spec), 1)
        self.assertIn((mass_spec[0][0], RDF.type, onto.SNS.input_specification), graph)
        self.assertIn(
            (mass_spec[0][0], onto.SNS.local_identifier, Literal("mass")), graph
        )
        self.assertIn(
            (mass_spec[0][0], onto.SNS.has_parameter_position, Literal(0)), graph
        )

        # Check the output specification
        query = sparql_prefixes + """
            SELECT ?output WHERE {
              ?output a ?bnode .
              ?bnode a owl:Restriction .
              ?bnode owl:onProperty iao:0000136 .
              ?bnode owl:allValuesFrom pmdco:0020142 .
            }"""
        output_spec = list(graph.query(query))
        self.assertEqual(len(output_spec), 1)
        self.assertIn(
            (output_spec[0][0], RDF.type, onto.SNS.output_specification), graph
        )
        self.assertIn(
            (output_spec[0][0], onto.SNS.local_identifier, Literal("kinetic_energy")),
            graph,
        )
        self.assertIn(
            (output_spec[0][0], onto.SNS.has_parameter_position, Literal(0)), graph
        )

    def test_run(self):
        wf_dict = my_kinetic_energy_workflow.run(distance=2, time=1, mass=4)
        g_run = onto.get_knowledge_graph(wf_dict)
        query = sparql_prefixes + """
        SELECT ?node ?value WHERE {
          ?bnode a ?node ;
            pmdco:0000006 ?value .
        }
        """
        results = list(g_run.query(query))
        for tag, value in zip(
            [
                "my_kinetic_energy_workflow-inputs-mass_data",
                "my_kinetic_energy_workflow-get_kinetic_energy_0-outputs-kinetic_energy_data",
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
        g_run_without_data = onto.get_knowledge_graph(wf_dict, remove_data=True)
        results = [d[0] for d in g_run_without_data.query(query)]
        for tag in [
            "my_kinetic_energy_workflow-get_kinetic_energy_0-outputs-output_data",
            "my_kinetic_energy_workflow-get_speed_0-outputs-speed_data",
        ]:
            with self.subTest(f"Checking value for {tag}"):
                self.assertFalse(any(str(r).endswith(tag) for r in results))

    def test_query_with_uri(self):
        wf_dict = my_kinetic_energy_workflow.run(distance=1.0, time=2.0, mass=3.0)
        graph = onto.get_knowledge_graph(wf_dict)

        query = sparql_prefixes + """
        SELECT ?distance_value ?e_value WHERE {
          ?distance_datanode pmdco:0000006 ?distance_value .
          ?distance_datanode obi:0001927 ?distance_bnode .
          ?distance_bnode a pmdco:0040001 .
          ?input_node ro:0000057 ?distance_datanode .
          ?workflow_node bfo:0000051 ?input_node .
          ?workflow_node bfo:0000051 ?output_node .
          ?output_node ro:0000057 ?e_datanode .
          ?e_bnode a pmdco:0020142 .
          ?e_datanode obi:0001927 ?e_bnode .
          ?e_datanode pmdco:0000006 ?e_value .
        }
        """
        data = [d.toPython() for d in list(graph.query(query))[0]]
        self.assertListEqual(data, [1.0, 0.375])

    def test_extract_dataclass(self):
        inp = Input(T=300.0, n=100)
        wf_dict = get_run_md.run(inp=inp, E=1.0)
        g = onto.get_knowledge_graph(wf_dict)
        g_dc = onto.extract_dataclass(g)
        with self.subTest("temperature data"):
            for s in g_dc.subjects(RDF.type, onto.BASE["get_run_md-inputs-inp_T_data"]):
                self.assertIn((s, onto.QUDT.hasUnit, UNIT["K"]), g_dc)
                self.assertIn((s, onto.SNS.has_value, Literal(300.0)), g_dc)

        with self.subTest("particle count data"):
            # Check that the particle count data has the correct value
            for s in g_dc.subjects(RDF.type, onto.BASE["get_run_md-inputs-inp_n_data"]):
                self.assertIn((s, onto.SNS.has_value, Literal(100)), g_dc)

        with self.subTest("energy output data"):
            for s in g_dc.subjects(
                RDF.type, onto.BASE["get_run_md-run_md_0-outputs-out_E_data"]
            ):
                self.assertIn((s, onto.QUDT.hasUnit, UNIT["EV"]), g_dc)
                self.assertIn((s, onto.SNS.has_value, Literal(1.0)), g_dc)
                self.assertIn((s, onto.OBI["0001927"], None), g_dc)

        with self.subTest("length output data"):
            for s in g_dc.subjects(
                RDF.type, onto.BASE["get_run_md-run_md_0-outputs-out_L_data"]
            ):
                self.assertIn((s, onto.QUDT.hasUnit, UNIT["NanoM"]), g_dc)
                self.assertIn((s, onto.SNS.has_value, Literal(2.0)), g_dc)

        speed_data = NewSpeedData(distance=1.0, time=2.0)
        query = sparql_prefixes + """
            SELECT ?distance_value ?e_value WHERE {
              ?distance_datanode pmdco:0000006 ?distance_value .
              ?distance_datanode obi:0001927 ?distance_bnode .
              ?distance_bnode a pmdco:0040001 .
              ?distance_datanode a ?distance_class .
              ?speed_data_class rdfs:subClassOf ?has_part_bnode .
              ?has_part_bnode owl:someValuesFrom ?distance_class .
              ?has_part_bnode owl:onProperty bfo:0000051 .
              ?has_part_bnode a owl:Restriction .
              ?speed_data_datanode a ?speed_data_class .
              ?input_node ro:0000057 ?speed_data_datanode .
              ?workflow_node bfo:0000051 ?input_node .
              ?workflow_node bfo:0000051 ?output_node .
              ?output_node ro:0000057 ?e_datanode .
              ?e_bnode a pmdco:0020142 .
              ?e_datanode obi:0001927 ?e_bnode .
              ?e_datanode pmdco:0000006 ?e_value .
            }"""

        wf_dict = workflow_with_dataclass.run(speed_data=speed_data, mass=3)
        g = onto.get_knowledge_graph(wf_dict, extract_dataclasses=False)
        self.assertListEqual(list(g.query(query)), [])
        sec = onto._units_to_uri("second")
        self.assertEqual(len(list(g.subjects(onto.QUDT.hasUnit, sec))), 0)
        g = onto.get_knowledge_graph(wf_dict, extract_dataclasses=True)
        self.assertGreater(len(list(g.subjects(onto.QUDT.hasUnit, sec))), 0)
        self.assertListEqual(
            [d.toPython() for d in list(g.query(query))[0]], [1, 0.375]
        )

    def test_function_to_knowledge_graph(self):
        g = onto.function_to_knowledge_graph(get_speed)
        g += onto.function_to_knowledge_graph(get_kinetic_energy)
        query = sparql_prefixes + """
            SELECT ?label ?import_path WHERE {
              ?function iao:0000136 ?bnode .
              ?function iao:0000235 ?f_name .
              ?f_name pmdco:0000006 ?label .
              ?f_name a pmdco:0000100 .
              ?bnode a ex:get_kinetic_energy .
              ?function iao:0000235 ?f_module .
              ?f_module a pmdco:0000101 .
              ?f_module pmdco:0000006 ?import_path .
            }"""
        self.assertEqual(
            "get_kinetic_energy", [row[0].toPython() for row in g.query(query)][0]
        )
        self.assertEqual(__name__, [row[1].toPython() for row in g.query(query)][0])
        query = sparql_prefixes + """
            SELECT ?label WHERE {
              ?function bfo:0000051 ?bnode .
              ?bnode pmdco:0000128 ?label .
            }"""
        g = onto.function_to_knowledge_graph(prepare_pizza)
        self.assertEqual(list(g.query(query))[0][0].toPython(), "output_0")

    def test_unhashable(self):
        uh = Unhashable()
        wf_dict = my_unhashable_inputs.run(uh=uh)
        with self.assertRaises(RuntimeError) as context:
            _ = onto.get_knowledge_graph(wf_dict)
            self.assertEqual(
                str(context.exception),
                "Failed to hash workflow data - use only hashable inputs or set hash_data=False",
            )

    def test_multiple_connections(self):

        # Check that the multiple connections for c do not cause error
        _ = onto.get_knowledge_graph(multiple_connection.get_semantikon_dict())

    def test_load_data(self):
        wf_dict = my_kinetic_energy_workflow.run(distance=1.0, time=2.0, mass=3.0)
        graph = onto.get_knowledge_graph(wf_dict, store_data=True, file_name="test")
        data = onto.load_data("test.h5")
        self.assertEqual(sorted(data.values()), [0.375, 0.5])
        for key, value in data.items():
            self.assertEqual(onto.load_data("test.h5", object_location=key), value)

        file_uri = Path("test.h5").absolute().as_uri()

        # Check that exactly one file data item node exists with the correct type and URL
        file_data_items = list(graph.subjects(RDF.type, onto.NFDI["0000027"]))
        self.assertEqual(
            len(file_data_items), 1, msg="Expected exactly one file data item node"
        )
        file_data_item = file_data_items[0]
        self.assertIn(
            (file_data_item, onto.NFDI["0001008"], Literal(file_uri)),
            graph,
            msg="File data item should have has_url pointing to the file URI",
        )

        # Check that the file data item has an HDF5 format spec as a part
        hdf5_format_nodes = list(graph.subjects(RDF.type, onto.EDAM["format_3590"]))
        self.assertEqual(
            len(hdf5_format_nodes), 1, msg="Expected exactly one HDF5 format spec node"
        )
        self.assertIn(
            (file_data_item, onto.BFO["0000051"], hdf5_format_nodes[0]),
            graph,
            msg="File data item should have HDF5 format spec as a part",
        )

        # Check that each data node with a value has a corresponding file part
        q = """SELECT DISTINCT ?data_node WHERE {
            ?data_node pmdco:0000006 ?value .
            ?data_node iao:0000235 ?hash_node .
            ?hash_node a iao:0020000 .
        }"""
        data_nodes = [row[0] for row in graph.query(q)]
        self.assertEqual(len(data_nodes), 2, msg="Expected two data nodes")
        for data_node in data_nodes:
            file_parts = [
                p
                for p in graph.objects(file_data_item, onto.BFO["0000051"])
                if (p, onto.IAO["0000136"], data_node) in graph
            ]
            self.assertEqual(
                len(file_parts),
                1,
                msg=f"Expected exactly one file part for data node {data_node}",
            )

        os.remove("test.h5")

    def test_nested_workflow(self):
        query = sparql_prefixes + """
        ASK {
            sns:T_wf_nested_triples rdfs:subClassOf ?bnode .
            ?bnode a owl:Restriction .
            ?bnode owl:onProperty bfo:0000051 .
            ?bnode owl:someValuesFrom sns:T_wf_nested_triples-wf_triples_0 .
        }"""
        g = onto.get_knowledge_graph(
            wf_nested_triples.get_semantikon_dict(), prefix="T"
        )
        self.assertTrue(g.query(query).askAnswer, msg=g.serialize())

    def test_inconsistent_workflow_and_child_node_inputs(self):
        wf_dict = swallow_pizza_painfully.get_semantikon_dict()
        g = onto.get_knowledge_graph(wf_dict)
        self.assertTrue(
            onto.validate_values(g)[0],
            msg=dedent("""
                As long as the knowledge graph does not know that EX.Meal and
                EX.Water are disjoint, it should not raise an error
            """),
        )
        g.add((EX.Meal, OWL.disjointWith, EX.Water))
        g.add((EX.Meal, RDF.type, OWL.Class))
        g.add((EX.Water, RDF.type, OWL.Class))
        self.assertFalse(
            onto.validate_values(g)[0],
            msg=dedent("""
                Once we add that EX.Meal and EX.Water are disjoint, the workflow
                should be inconsistent because it requires an individual to be
                both a Meal and a Water.
            """),
        )

    def test_inheritance_of_derives_from(self):
        g = onto.get_knowledge_graph(wf_triples.get_semantikon_dict())
        onto._inherit_properties(g)
        uri_node = list(g.subjects(RDF.type, EX.Something))[0]
        data_node = list(g.subjects(onto.SNS.specifies_value_of, uri_node))
        self.assertIn(
            "wf_triples-inputs-a_data",
            str(data_node[0]),
            msg=dedent(f"""
                Expected only the input data node specifying value of {uri_node}
                (which should end with 'wf_triples-inputs-a_data'), found
                {data_node}
                """),
        )


if __name__ == "__main__":
    unittest.main()
