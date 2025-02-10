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
    get_triples,
    _inherit_properties,
    validate_values,
)
from dataclasses import dataclass


EX = Namespace("http://example.org/")


def get_speed():
    return {
        "speed": {
            "inputs": {
                "distance": {
                    "units": "meter",
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                    "value": 10.0,
                    "connection": None,
                },
                "time": {
                    "units": "second",
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                    "value": 2.0,
                    "connection": None,
                },
            },
            "outputs": {
                "speed": {
                    "units": "meter/second",
                    "label": None,
                    "triples": (
                        (
                            EX.somehowRelatedTo,
                            "inputs.time",
                        ),
                        (
                            EX.subject,
                            EX.predicate,
                            EX.object,
                        ),
                        (
                            EX.subject,
                            EX.predicate,
                            None,
                        ),
                        (
                            None,
                            EX.predicate,
                            EX.object,
                        ),
                    ),
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                    "value": 5.0,
                }
            },
            "function": {"label": "calculate_speed"},
            "label": "speed",
        },
        "workflow_label": "speed",
    }


def get_correct_analysis():
    return {
        "addition": {
            "inputs": {
                "a": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                    "value": 1.0,
                    "connection": None,
                },
                "b": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                    "value": 2.0,
                    "connection": None,
                },
            },
            "outputs": {
                "result": {
                    "units": None,
                    "label": None,
                    "triples": (
                        EX.HasOperation,
                        EX.Addition,
                    ),
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                }
            },
            "function": {
                "label": "add",
                "triples": None,
                "uri": EX.Addition,
                "restrictions": None,
                "use_list": True,
            },
            "label": "addition",
        },
        "multiply": {
            "inputs": {
                "a": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                    "connection": "addition.outputs.result",
                },
                "b": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                    "value": 3.0,
                    "connection": None,
                },
            },
            "outputs": {
                "result": {
                    "units": None,
                    "label": None,
                    "triples": (
                        (
                            EX.HasOperation,
                            EX.Multiplication,
                        ),
                        (
                            PNS.inheritsPropertiesFrom,
                            "inputs.a",
                        ),
                    ),
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                }
            },
            "function": {"label": "multiply"},
            "label": "multiply",
        },
        "analysis": {
            "inputs": {
                "a": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": (
                        (
                            OWL.onProperty,
                            EX.HasOperation,
                        ),
                        (
                            OWL.someValuesFrom,
                            EX.Addition,
                        ),
                    ),
                    "dtype": float,
                    "connection": "multiply.outputs.result",
                }
            },
            "outputs": {
                "result": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                }
            },
            "function": {"label": "correct_analysis"},
            "label": "analysis",
        },
        "workflow_label": "correct_analysis",
    }


def get_wrong_analysis():
    return {
        "addition": {
            "inputs": {
                "a": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                    "value": 1.0,
                    "connection": None,
                },
                "b": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                    "value": 2.0,
                    "connection": None,
                },
            },
            "outputs": {
                "result": {
                    "units": None,
                    "label": None,
                    "triples": (
                        EX.HasOperation,
                        EX.Addition,
                    ),
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                }
            },
            "function": {
                "label": "add",
                "triples": None,
                "uri": EX.Addition,
                "restrictions": None,
                "use_list": True,
            },
            "label": "addition",
        },
        "multiply": {
            "inputs": {
                "a": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                    "connection": "addition.outputs.result",
                },
                "b": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                    "value": 3.0,
                    "connection": None,
                },
            },
            "outputs": {
                "result": {
                    "units": None,
                    "label": None,
                    "triples": (
                        (
                            EX.HasOperation,
                            EX.Multiplication,
                        ),
                        (
                            PNS.inheritsPropertiesFrom,
                            "inputs.a",
                        ),
                    ),
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                }
            },
            "function": {"label": "multiply"},
            "label": "multiply",
        },
        "analysis": {
            "inputs": {
                "a": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": (
                        (
                            OWL.onProperty,
                            EX.HasOperation,
                        ),
                        (
                            OWL.someValuesFrom,
                            EX.Division,
                        ),
                    ),
                    "dtype": float,
                    "connection": "multiply.outputs.result",
                }
            },
            "outputs": {
                "result": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": float,
                }
            },
            "function": {"label": "wrong_analysis"},
            "label": "analysis",
        },
        "workflow_label": "wrong_analysis",
    }


class TestOntology(unittest.TestCase):
    def test_units_with_sparql(self):
        graph = get_knowledge_graph(get_speed())
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
        self.assertEqual(len(results), 3)
        result_list = [row[0].value for row in graph.query(query)]
        self.assertEqual(sorted(result_list), [2.0, 5.0, 10.0])

    def test_triples(self):
        data = get_speed()["speed"]
        graph = get_triples(data=data)
        subj = URIRef("http://example.org/subject")
        obj = URIRef("http://example.org/object")
        label = URIRef("speed.outputs.speed")
        self.assertGreater(
            len(list(graph.triples((None, PNS.hasUnits, URIRef("meter/second"))))), 0
        )
        ex_triple = (None, EX.somehowRelatedTo, URIRef("speed.inputs.time"))
        self.assertTrue(
            ex_triple in graph,
            msg=f"Triple {ex_triple} not found {graph.serialize(format='turtle')}",
        )
        self.assertEqual(
            len(list(graph.triples((EX.subject, EX.predicate, EX.object)))), 1
        )
        self.assertTrue((subj, EX.predicate, label) in graph)
        self.assertTrue((label, EX.predicate, obj) in graph)

    def test_correct_analysis(self):
        graph = get_knowledge_graph(get_correct_analysis())
        _inherit_properties(graph)
        DeductiveClosure(OWLRL_Semantics).expand(graph)
        self.assertEqual(len(validate_values(graph)), 0)
        graph = get_knowledge_graph(get_wrong_analysis())
        _inherit_properties(graph)
        DeductiveClosure(OWLRL_Semantics).expand(graph)
        self.assertEqual(len(validate_values(graph)), 1)


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


def get_run_md():
    inp = Input(T=300.0, n=100)
    inp.parameters.a = 1
    return {
        "node": {
            "inputs": {
                "inp": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": Input,
                    "value": inp,
                    "connection": None,
                }
            },
            "outputs": {
                "out": {
                    "units": None,
                    "label": None,
                    "triples": None,
                    "uri": None,
                    "shape": None,
                    "restrictions": None,
                    "dtype": Output,
                    "value": Output(E=1.0, L=2.0),
                }
            },
            "function": {"label": "run_md"},
            "label": "node",
        },
        "workflow_label": "my_wf",
    }


class TestDataclass(unittest.TestCase):
    def test_dataclass(self):
        graph = get_knowledge_graph(get_run_md())
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
