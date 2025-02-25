import unittest
from rdflib import Graph, OWL, Namespace, URIRef, Literal, RDF, RDFS
from owlrl import DeductiveClosure, OWLRL_Semantics
from semantikon.typing import u
from semantikon.converter import (
    parse_input_args,
    parse_output_args,
    parse_metadata,
    get_function_dict,
)
from semantikon.ontology import (
    get_knowledge_graph,
    PNS,
    _inherit_properties,
    validate_values,
)
from semantikon.workflow import workflow
from dataclasses import dataclass


EX = Namespace("http://example.org/")


def calculate_speed(
    distance: u(float, units="meter") = 10.0,
    time: u(float, units="second") = 2.0,
) -> u(
    float,
    units="meter/second",
    triples=(
        (EX.somehowRelatedTo, "inputs.time"),
        (EX.subject, EX.predicate, EX.object),
        (EX.subject, EX.predicate, None),
        (None, EX.predicate, EX.object),
    ),
):
    return distance / time


@workflow
def get_speed(distance=10.0, time=2.0):
    speed = calculate_speed(distance, time)
    return speed


get_speed.run()


@u(uri=EX.Addition)
def add(a: float, b: float) -> u(float, triples=(EX.HasOperation, EX.Addition)):
    return a + b


def multiply(a: float, b: float) -> u(
    float,
    triples=(
        (EX.HasOperation, EX.Multiplication),
        (PNS.inheritsPropertiesFrom, "inputs.a"),
    ),
):
    return a * b


def correct_analysis(
    a: u(
        float,
        restrictions=(
            (OWL.onProperty, EX.HasOperation),
            (OWL.someValuesFrom, EX.Addition),
        ),
    ),
) -> float:
    return a


def wrong_analysis(
    a: u(
        float,
        restrictions=(
            (OWL.onProperty, EX.HasOperation),
            (OWL.someValuesFrom, EX.Division),
        ),
    ),
) -> float:
    return a


def multiple_outputs(a: int = 1, b: int = 2) -> tuple[int, int]:
    return a, b


def add_one(a: int):
    result = a + 1
    return result


def add_two(b=10) -> int:
    result = b + 2
    return result


@workflow
def add_three(c: int) -> int:
    one = add_one(a=c)
    w = add_two(b=one)
    return w


def create_vacancy(
    structure: str,
) -> u(
    str,
    triples=(
        (PNS.inheritsPropertiesFrom, "inputs.structure"),
        (EX.hasDefect, EX.vacancy),
    ),
    cancel=(EX.hasState, EX.relaxed),
):
    return structure


def relax_structure(
    structure: str,
) -> u(
    str,
    triples=(
        (PNS.inheritsPropertiesFrom, "inputs.structure"),
        (EX.hasState, EX.relaxed),
    ),
):
    return structure


def get_vacancy_formation_energy(
    structure: u(
        str,
        restrictions=(
            ((OWL.onProperty, EX.hasDefect), (OWL.someValuesFrom, EX.vacancy)),
            ((OWL.onProperty, EX.hasState), (OWL.someValuesFrom, EX.relaxed)),
        ),
    ),
):
    return len(structure)


@workflow
def get_correct_analysis(a=1.0, b=2.0, c=3.0):
    d = add(a=a, b=b)
    m = multiply(a=d, b=c)
    analysis = correct_analysis(a=m)
    return analysis


@workflow
def get_wrong_analysis(a=1.0, b=2.0, c=3.0):
    d = add(a=a, b=b)
    m = multiply(a=d, b=c)
    analysis = wrong_analysis(a=m)
    return analysis


@workflow
def get_macro(c=1):
    w = add_three(c=c)
    result = add_one(a=w)
    return result


@workflow
def get_wrong_order(structure="abc"):
    relaxed = relax_structure(structure=structure)
    vac = create_vacancy(structure=relaxed)
    energy = get_vacancy_formation_energy(structure=vac)
    return energy


class TestOntology(unittest.TestCase):
    def test_units_with_sparql(self):
        graph = get_knowledge_graph(get_speed._semantikon_workflow)
        query_txt = [
            "PREFIX ex: <http://example.org/>",
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            f"PREFIX pns: <{PNS.BASE}>",
            "SELECT DISTINCT ?speed ?units",
            "WHERE {",
            "    ?output pns:hasValue ?output_tag .",
            "    ?output_tag rdf:value ?speed .",
            "    ?output_tag pns:hasUnits ?units .",
            "}",
        ]
        query = "\n".join(query_txt)
        results = graph.query(query)
        self.assertEqual(
            len(results), 3, msg=f"Results: {results.serialize(format='txt').decode()}"
        )
        result_list = [row[0].value for row in graph.query(query)]
        self.assertEqual(sorted(result_list), [2.0, 5.0, 10.0])

    def test_triples(self):
        graph = get_knowledge_graph(get_speed._semantikon_workflow)
        subj = URIRef("http://example.org/subject")
        obj = URIRef("http://example.org/object")
        label = URIRef("get_speed.calculate_speed_0.outputs.output")
        self.assertGreater(
            len(list(graph.triples((None, PNS.hasUnits, URIRef("meter/second"))))), 0
        )
        ex_triple = (
            None,
            EX.somehowRelatedTo,
            URIRef("get_speed.calculate_speed_0.inputs.time"),
        )
        self.assertTrue(
            ex_triple in graph,
            msg=f"Triple {ex_triple} not found {graph.serialize(format='turtle')}",
        )
        self.assertTrue((EX.subject, EX.predicate, EX.object) in graph)
        self.assertTrue((subj, EX.predicate, label) in graph)
        self.assertTrue((label, EX.predicate, obj) in graph)

    def test_correct_analysis(self):
        graph = get_knowledge_graph(get_correct_analysis._semantikon_workflow)
        missing_triples = validate_values(graph)
        self.assertEqual(
            len(missing_triples),
            0,
            msg=f"{missing_triples} not found in {graph.serialize()}",
        )
        graph = get_knowledge_graph(get_wrong_analysis._semantikon_workflow)
        self.assertEqual(len(validate_values(graph)), 1)

    def test_macro(self):
        graph = get_knowledge_graph(get_macro())
        subj = list(
            graph.subjects(PNS.hasValue, URIRef("my_wf.three.two.outputs.result.value"))
        )
        self.assertEqual(len(subj), 3)
        prefix = "my_wf"
        for ii, tag in enumerate(
            ["three.two.outputs.result", "three.outputs.w", "four.inputs.a"]
        ):
            with self.subTest(i=ii):
                self.assertIn(
                    URIRef(prefix + "." + tag), subj, msg=f"{tag} not in {subj}"
                )
        same_as = [(str(g[0]), str(g[1])) for g in graph.subject_objects(OWL.sameAs)]
        sub_obj = [
            ("my_wf.three.one.inputs.a", "my_wf.three.inputs.c"),
            ("my_wf.three.outputs.w", "my_wf.three.two.outputs.result"),
        ]
        self.assertEqual(len(same_as), 2)
        for ii, pair in enumerate(sub_obj):
            with self.subTest(i=ii):
                self.assertIn(pair, same_as)

    def test_wrong_order(self):
        graph = get_knowledge_graph(get_wrong_order._semantikon_workflow)
        missing_triples = [[str(gg) for gg in g] for g in validate_values(graph)]
        self.assertEqual(
            missing_triples,
            [
                [
                    "get_wrong_order.get_vacancy_formation_energy_0.inputs.structure",
                    "http://example.org/hasState",
                    "http://example.org/relaxed",
                ]
            ],
        )


@dataclass
class Input:
    T: u(float, units="kelvin")
    n: int

    @dataclass
    class parameters:
        a: int = 2

    class not_dataclass:
        b: int = 3


@dataclass
class Output:
    E: u(float, units="electron_volt")
    L: u(float, units="angstrom")


def run_md(inp: Input) -> Output:
    out = Output(E=1.0, L=2.0)
    return out


def get_run_md_dict():
    inp = Input(T=300.0, n=100)
    inp.parameters.a = 1
    return {
        "inputs": {"node__inp": {"value": inp, "type_hint": Input}},
        "outputs": {"node__out": {"value": Output(E=1.0, L=2.0), "type_hint": Output}},
        "nodes": {
            "node": {
                "inputs": {"inp": {"value": inp, "type_hint": Input}},
                "outputs": {
                    "out": {"value": Output(E=1.0, L=2.0), "type_hint": Output}
                },
                "function": run_md,
            }
        },
        "data_edges": [],
        "label": "my_wf",
    }


class TestDataclass(unittest.TestCase):
    def test_dataclass(self):
        graph = get_knowledge_graph(get_run_md_dict())
        i_txt = "my_wf.node.inputs.inp"
        o_txt = "my_wf.node.outputs.out"
        triples = (
            (URIRef(f"{i_txt}.n.value"), RDFS.subClassOf, URIRef(f"{i_txt}.value")),
            (URIRef(f"{i_txt}.n.value"), RDF.value, Literal(100)),
            (URIRef(f"{i_txt}.parameters.a.value"), RDF.value, Literal(1)),
            (URIRef(o_txt), PNS.hasValue, URIRef(f"{o_txt}.E.value")),
        )
        s = graph.serialize(format="turtle")
        for ii, triple in enumerate(triples):
            with self.subTest(i=ii):
                self.assertEqual(
                    len(list(graph.triples(triple))),
                    1,
                    msg=f"{triple} not found in {s}",
                )
        self.assertIsNone(graph.value(URIRef(f"{i_txt}.not_dataclass.b.value")))


if __name__ == "__main__":
    unittest.main()
