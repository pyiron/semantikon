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
    )
) -> float:
    return a


def wrong_analysis(
    a: u(
        float,
        restrictions=(
            (OWL.onProperty, EX.HasOperation),
            (OWL.someValuesFrom, EX.Division),
        ),
    )
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


def add_three(macro, c: int) -> int:
    macro.one = add_one(a=c)
    macro.two = add_two(b=macro.one)
    w = macro.two
    return w


def get_speed_dict():
    return {
        "inputs": {
            "speed__distance": {
                "default": 10.0,
                "value": 10.0,
                "type_hint": u(float, units="meter"),
            },
            "speed__time": {
                "default": 2.0,
                "value": 2.0,
                "type_hint": u(float, units="second"),
            },
        },
        "outputs": {
            "speed__speed": {
                "type_hint": u(
                    float,
                    units="meter/second",
                    triples=(
                        (EX.somehowRelatedTo, "inputs.time"),
                        (EX.subject, EX.predicate, EX.object),
                        (EX.subject, EX.predicate, None),
                        (None, EX.predicate, EX.object),
                    ),
                ),
            }
        },
        "nodes": {
            "speed": {
                "inputs": {
                    "distance": {
                        "default": 10.0,
                        "value": 10.0,
                        "type_hint": u(float, units="meter"),
                    },
                    "time": {
                        "default": 2.0,
                        "value": 2.0,
                        "type_hint": u(float, units="second"),
                    },
                },
                "outputs": {
                    "speed": {
                        "type_hint": u(
                            float,
                            units="meter/second",
                            triples=(
                                (EX.somehowRelatedTo, "inputs.time"),
                                (EX.subject, EX.predicate, EX.object),
                                (EX.subject, EX.predicate, None),
                                (None, EX.predicate, EX.object),
                            ),
                        ),
                    }
                },
                "function": calculate_speed,
            }
        },
        "data_edges": [],
        "label": "speed",
    }


def get_correct_analysis_dict():
    return {
        "inputs": {
            "addition__a": {"value": 1.0, "type_hint": float},
            "addition__b": {"value": 2.0, "type_hint": float},
            "multiply__b": {"value": 3.0, "type_hint": float},
        },
        "outputs": {"analysis__result": {"type_hint": float}},
        "nodes": {
            "addition": {
                "inputs": {
                    "a": {"value": 1.0, "type_hint": float},
                    "b": {"value": 2.0, "type_hint": float},
                },
                "outputs": {
                    "result": {
                        "type_hint": u(float, triples=(EX.HasOperation, EX.Addition)),
                    }
                },
                "function": add,
            },
            "multiply": {
                "inputs": {
                    "a": {"type_hint": float},
                    "b": {"value": 3.0, "type_hint": float},
                },
                "outputs": {
                    "result": {
                        "type_hint": u(
                            float,
                            triples=(
                                (EX.HasOperation, EX.Multiplication),
                                (PNS.inheritsPropertiesFrom, "inputs.a"),
                            ),
                        ),
                    }
                },
                "function": multiply,
            },
            "analysis": {
                "inputs": {
                    "a": {
                        "type_hint": u(
                            float,
                            restrictions=(
                                (OWL.onProperty, EX.HasOperation),
                                (OWL.someValuesFrom, EX.Addition),
                            ),
                        )
                    }
                },
                "outputs": {"result": {"type_hint": float}},
                "function": correct_analysis,
            },
        },
        "data_edges": [
            ("addition.outputs.result", "multiply.inputs.a"),
            ("multiply.outputs.result", "analysis.inputs.a"),
        ],
        "label": "correct_analysis",
    }


def get_wrong_analysis_dict():
    return {
        "inputs": {
            "addition__a": {"value": 1.0, "type_hint": float},
            "addition__b": {"value": 2.0, "type_hint": float},
            "multiply__b": {"value": 3.0, "type_hint": float},
        },
        "outputs": {"analysis__result": {"type_hint": float}},
        "nodes": {
            "addition": {
                "inputs": {
                    "a": {"value": 1.0, "type_hint": float},
                    "b": {"value": 2.0, "type_hint": float},
                },
                "outputs": {
                    "result": {
                        "type_hint": u(float, triples=(EX.HasOperation, EX.Addition))
                    }
                },
                "function": add,
            },
            "multiply": {
                "inputs": {
                    "a": {"type_hint": float},
                    "b": {"value": 3.0, "type_hint": float},
                },
                "outputs": {
                    "result": {
                        "type_hint": u(
                            float,
                            triples=(
                                (EX.HasOperation, EX.Multiplication),
                                (PNS.inheritsPropertiesFrom, "inputs.a"),
                            ),
                        )
                    }
                },
                "function": multiply,
            },
            "analysis": {
                "inputs": {
                    "a": {
                        "type_hint": u(
                            float,
                            restrictions=(
                                (OWL.onProperty, EX.HasOperation),
                                (OWL.someValuesFrom, EX.Division),
                            ),
                        )
                    }
                },
                "outputs": {"result": {"type_hint": float}},
                "function": wrong_analysis,
            },
        },
        "data_edges": [
            ("addition.outputs.result", "multiply.inputs.a"),
            ("multiply.outputs.result", "analysis.inputs.a"),
        ],
        "label": "wrong_analysis",
    }


def get_macro():
    return {
        "inputs": {"three__c": {"value": 1, "type_hint": int}},
        "outputs": {"four__result": {"value": 5}},
        "nodes": {
            "three": {
                "inputs": {"three__c": {"value": 1, "type_hint": int}},
                "outputs": {"three__w": {"value": 4, "type_hint": int}},
                "nodes": {
                    "one": {
                        "inputs": {"a": {"value": 1, "type_hint": int}},
                        "outputs": {"result": {"value": 2}},
                        "function": add_one,
                    },
                    "two": {
                        "inputs": {"b": {"default": 10, "value": 2}},
                        "outputs": {"result": {"value": 4, "type_hint": int}},
                        "function": add_two,
                    },
                },
                "data_edges": [
                    ("inputs.three__c", "one.inputs.a"),
                    ("one.outputs.result", "two.inputs.b"),
                    ("two.outputs.result", "outputs.three__w"),
                ],
                "label": "three",
            },
            "four": {
                "inputs": {"a": {"value": 4, "type_hint": int}},
                "outputs": {"result": {"value": 5}},
                "function": add_one,
            },
        },
        "data_edges": [("three.outputs.w", "four.inputs.a")],
        "label": "my_wf",
    }


class TestOntology(unittest.TestCase):
    def test_units_with_sparql(self):
        graph = get_knowledge_graph(get_speed_dict())
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
            len(results), 2, msg=f"Results: {results.serialize(format='txt').decode()}"
        )
        result_list = [row[0].value for row in graph.query(query)]
        self.assertEqual(sorted(result_list), [2.0, 10.0])

    def test_triples(self):
        data = get_speed_dict()
        graph = get_knowledge_graph(data)
        subj = URIRef("http://example.org/subject")
        obj = URIRef("http://example.org/object")
        label = URIRef("speed.speed.outputs.speed")
        self.assertGreater(
            len(list(graph.triples((None, PNS.hasUnits, URIRef("meter/second"))))), 0
        )
        ex_triple = (None, EX.somehowRelatedTo, URIRef("speed.speed.inputs.time"))
        self.assertTrue(
            ex_triple in graph,
            msg=f"Triple {ex_triple} not found {graph.serialize(format='turtle')}",
        )
        self.assertTrue((EX.subject, EX.predicate, EX.object) in graph)
        self.assertTrue((subj, EX.predicate, label) in graph)
        self.assertTrue((label, EX.predicate, obj) in graph)

    def test_correct_analysis(self):
        graph = get_knowledge_graph(get_correct_analysis_dict())
        _inherit_properties(graph)
        DeductiveClosure(OWLRL_Semantics).expand(graph)
        missing_triples = validate_values(graph)
        self.assertEqual(
            len(missing_triples),
            0,
            msg=f"{missing_triples} not found in {graph.serialize()}",
        )
        graph = get_knowledge_graph(get_wrong_analysis_dict())
        _inherit_properties(graph)
        DeductiveClosure(OWLRL_Semantics).expand(graph)
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
