import unittest
from rdflib import Graph, OWL, Namespace, URIRef
from semantikon.typing import u
from semantikon.converter import (
    parse_input_args,
    parse_output_args,
    parse_metadata,
    get_function_dict,
)
from semantikon.ontology import get_knowledge_graph, PNS, get_triples


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


if __name__ == "__main__":
    unittest.main()
