import unittest
from dataclasses import dataclass
from textwrap import dedent

from graphviz import Digraph
from pyshacl import validate
from rdflib import OWL, PROV, RDF, RDFS, SH, Graph, Literal, Namespace, URIRef
from rdflib.compare import graph_diff

from semantikon.metadata import u
from semantikon.ontology import (
    NS,
    SNS,
    _parse_cancel,
    dataclass_to_knowledge_graph,
    get_knowledge_graph,
    serialize_data,
    validate_values,
)
from semantikon.visualize import visualize
from semantikon.workflow import workflow

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


@u(uri=EX.Addition, triples=("inputs.a", PROV.wasGeneratedBy, None))
def add(a: float, b: float) -> u(float, triples=(EX.HasOperation, EX.Addition)):
    return a + b


def multiply(a: float, b: float) -> u(
    float,
    triples=(EX.HasOperation, EX.Multiplication),
    derived_from="inputs.a",
):
    return a * b


def correct_analysis(a: u(float, triples=(EX.HasOperation, EX.Addition))) -> float:
    return a


def wrong_analysis(a: u(float, triples=(EX.HasOperation, EX.Division))) -> float:
    return a


def correct_analysis_owl(
    a: u(
        float,
        restrictions=(
            (OWL.onProperty, EX.HasOperation),
            (OWL.someValuesFrom, EX.Addition),
        ),
    ),
) -> float:
    return a


def wrong_analysis_owl(
    a: u(
        float,
        restrictions=(
            (OWL.onProperty, EX.HasOperation),
            (OWL.someValuesFrom, EX.Division),
        ),
    ),
) -> float:
    return a


def correct_analysis_sh(
    a: u(
        float,
        restrictions=(
            (SH.path, EX.HasOperation),
            (SH.hasValue, EX.Addition),
        ),
    ),
) -> float:
    return a


def wrong_analysis_sh(
    a: u(
        float,
        restrictions=(
            (SH.path, EX.HasOperation),
            (SH.hasValue, EX.Division),
        ),
    ),
) -> float:
    return a


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
    triples=(EX.hasDefect, EX.vacancy),
    derived_from="inputs.structure",
    cancel=(EX.hasState, EX.relaxed),
):
    return structure


def relax_structure(
    structure: str,
) -> u(
    str,
    triples=(EX.hasState, EX.relaxed),
    derived_from="inputs.structure",
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


def get_vacancy_formation_energy_fulfills(
    structure: u(
        str,
        triples=(
            (EX.hasDefect, EX.vacancy),
            (EX.hasState, EX.relaxed),
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
def get_correct_analysis_owl(a=1.0, b=2.0, c=3.0):
    d = add(a=a, b=b)
    m = multiply(a=d, b=c)
    analysis = correct_analysis_owl(a=m)
    return analysis


@workflow
def get_wrong_analysis_owl(a=1.0, b=2.0, c=3.0):
    d = add(a=a, b=b)
    m = multiply(a=d, b=c)
    analysis = wrong_analysis_owl(a=m)
    return analysis


@workflow
def get_correct_analysis_sh(a=1.0, b=2.0, c=3.0):
    d = add(a=a, b=b)
    m = multiply(a=d, b=c)
    analysis = correct_analysis_sh(a=m)
    return analysis


@workflow
def get_wrong_analysis_sh(a=1.0, b=2.0, c=3.0):
    d = add(a=a, b=b)
    m = multiply(a=d, b=c)
    analysis = wrong_analysis_sh(a=m)
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


@workflow
def get_right_order_for_fulfillment(structure="abc"):
    vac = create_vacancy(structure=structure)
    relaxed = relax_structure(structure=vac)
    energy = get_vacancy_formation_energy_fulfills(structure=relaxed)
    return energy


@workflow
def get_wrong_order_for_fulfillment(structure="abc"):
    relaxed = relax_structure(structure=structure)
    vac = create_vacancy(structure=relaxed)
    energy = get_vacancy_formation_energy_fulfills(structure=vac)
    return energy


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


class Clothes:
    pass


def machine_wash(
    clothes: u(
        Clothes,
        uri=EX.Garment,
        triples=(EX.hasProperty, EX.MachineWashable),
    ),
) -> u(
    Clothes,
    uri=EX.SomethingElse,
    triples=(EX.hasProperty, EX.Cleaned),
    derived_from="inputs.clothes",
):
    return clothes


@workflow
def clothes_wf(
    clothes: u(
        Clothes,
        uri=EX.Garment,
        triples=(EX.hasProperty, EX.MachineWashable),
    ),
):
    washed = machine_wash(clothes)
    return washed


def add_onetology(x: u(int, uri=EX.Input)) -> u(int, uri=EX.Output):
    y = x + 1
    return y


@workflow
def matching_wrapper(x_outer: u(int, uri=EX.Input)) -> u(int, uri=EX.Output):
    add = add_onetology(x_outer)
    return add


@workflow
def mismatching_input(x_outer: u(int, uri=EX.NotInput)) -> u(int, uri=EX.Output):
    add = add_onetology(x_outer)
    return add


@workflow
def mismatching_output(x_outer: u(int, uri=EX.Input)) -> u(int, uri=EX.NotOutput):
    add = add_onetology(x_outer)
    return add


def dont_add_onetology(x: u(int, uri=EX.NotOutput)) -> u(int, uri=EX.NotOutput):
    y = x
    return y


@workflow
def mismatching_peers(x_outer: u(int, uri=EX.Input)) -> u(int, uri=EX.NotOutput):
    add = add_onetology(x_outer)
    dont_add = dont_add_onetology(add)
    return dont_add


class Foo: ...


def upstream(
    foo: u(Foo, uri=EX.Foo, triples=(EX.alsoHas, EX.Bar)),
) -> u(Foo, derived_from="inputs.foo"):
    return foo


def downstream(
    foo: u(
        Foo,
        uri=EX.Foo,
        restrictions=((OWL.onProperty, EX.alsoHas), (OWL.someValuesFrom, EX.Bar)),
    ),
) -> u(Foo, derived_from="inputs.foo"):
    return foo


@workflow
def single(foo: u(Foo, uri=EX.Foo)) -> u(Foo, derived_from="inputs.foo"):
    up = upstream(foo)
    dn = downstream(up)
    return dn


@workflow
def chain(foo: u(Foo, uri=EX.Foo)) -> u(Foo, derived_from="inputs.foo"):
    up = upstream(foo)
    dn1 = downstream(up)
    dn2 = downstream(dn1)
    return dn2


class TestOntology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None

    def test_units_with_sparql(self):
        graph = get_knowledge_graph(get_speed.run())
        query_txt = [
            "PREFIX ex: <http://example.org/>",
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
            f"PREFIX pns: <{SNS.BASE}>",
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
        graph = get_knowledge_graph(get_speed.run())
        subj = URIRef("http://example.org/subject")
        obj = URIRef("http://example.org/object")
        label = URIRef("get_speed.calculate_speed_0.outputs.output")
        self.assertGreater(
            len(
                list(
                    graph.subjects(
                        SNS.hasUnits, URIRef("http://qudt.org/vocab/unit/M-PER-SEC")
                    )
                )
            ),
            0,
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
        t = validate_values(graph)
        self.assertFalse(
            t["broken_promises"],
            msg=f"{t['broken_promises']} expected to be empty in {graph.serialize()}",
        )
        graph = get_knowledge_graph(get_wrong_analysis_owl._semantikon_workflow)
        self.assertEqual(len(validate_values(graph)["broken_promises"]), 1)

    def test_correct_analysis_owl(self):
        graph = get_knowledge_graph(get_correct_analysis_owl._semantikon_workflow)
        t = validate_values(graph)
        self.assertEqual(
            t["missing_triples"],
            [],
            msg=f"{t} missing in {graph.serialize()}",
        )
        graph = get_knowledge_graph(get_wrong_analysis_owl._semantikon_workflow)
        self.assertEqual(len(validate_values(graph)["missing_triples"]), 1)

    def test_correct_analysis_sh(self):
        graph = get_knowledge_graph(get_correct_analysis_sh._semantikon_workflow)
        self.assertTrue(validate(graph)[0])
        graph = get_knowledge_graph(get_wrong_analysis_sh._semantikon_workflow)
        self.assertFalse(validate(graph)[0])

    def test_valid_connections(self):
        graph = get_knowledge_graph(eat_pizza._semantikon_workflow)
        self.assertEqual(len(validate_values(graph)["incompatible_connections"]), 1)
        graph.add((EX.Pizza, RDFS.subClassOf, EX.Meal))
        self.assertEqual(validate_values(graph)["incompatible_connections"], [])

    def test_workflow_edge_validation(self):
        with self.subTest("Matching"):
            graph = get_knowledge_graph(matching_wrapper._semantikon_workflow)
            result = validate_values(graph)
            self.assertEqual(result["missing_triples"], [])
            self.assertEqual(result["incompatible_connections"], [])

        with self.subTest("Mismatching input"):
            graph = get_knowledge_graph(mismatching_input._semantikon_workflow)
            result = validate_values(graph)
            incompatible = [
                (
                    str(a),
                    str(b),
                    [str(x) for x in expected],
                    [str(x) for x in provided],
                )
                for (a, b, expected, provided) in result["incompatible_connections"]
            ]
            self.assertEqual(
                incompatible,
                [
                    (
                        "mismatching_input.add_onetology_0.inputs.x",
                        "mismatching_input.inputs.x_outer",
                        ["http://example.org/Input", "http://example.org/NotInput"],
                        ["http://example.org/NotInput"],
                    )
                ],
            )

        with self.subTest("Mismatching output"):
            graph = get_knowledge_graph(mismatching_output._semantikon_workflow)
            result = validate_values(graph)
            incompatible = [
                (
                    str(a),
                    str(b),
                    [str(x) for x in expected],
                    [str(x) for x in provided],
                )
                for (a, b, expected, provided) in result["incompatible_connections"]
            ]
            self.assertEqual(
                incompatible,
                [
                    (
                        "mismatching_output.outputs.add",
                        "mismatching_output.add_onetology_0.outputs.y",
                        ["http://example.org/NotOutput", "http://example.org/Output"],
                        ["http://example.org/Output"],
                    )
                ],
            )

        with self.subTest("Mismatching peers"):
            graph = get_knowledge_graph(mismatching_peers._semantikon_workflow)
            result = validate_values(graph)
            incompatible = [
                (
                    str(a),
                    str(b),
                    [str(x) for x in expected],
                    [str(x) for x in provided],
                )
                for (a, b, expected, provided) in result["incompatible_connections"]
            ]
            self.assertEqual(
                incompatible,
                [
                    (
                        "mismatching_peers.dont_add_onetology_0.inputs.x",
                        "mismatching_peers.add_onetology_0.outputs.y",
                        ["http://example.org/NotOutput", "http://example.org/Output"],
                        ["http://example.org/Output"],
                    )
                ],
            )

        with self.subTest("Externally informed peers"):
            context = Graph()
            context.add((EX.Output, RDFS.subClassOf, EX.NotOutput))
            graph = get_knowledge_graph(
                wf_dict=mismatching_peers._semantikon_workflow,
                graph=context,
            )
            result = validate_values(graph)
            self.assertEqual(result["missing_triples"], [])
            self.assertEqual(result["incompatible_connections"], [])

        with self.subTest("Wrongly informed peers"):
            context = Graph()
            context.add((EX.NotOutput, RDFS.subClassOf, EX.Output))  # Reversed
            # Now we're saying the downstream is expecting a subclass of the upstream,
            # which the upstream base class is _not_ guaranteeing
            graph = get_knowledge_graph(
                wf_dict=mismatching_peers._semantikon_workflow,
                graph=context,
            )
            result = validate_values(graph)
            incompatible = [
                (
                    str(a),
                    str(b),
                    [str(x) for x in expected],
                    [str(x) for x in provided],
                )
                for (a, b, expected, provided) in result["incompatible_connections"]
            ]
            self.assertEqual(
                incompatible,
                [
                    (
                        "mismatching_peers.dont_add_onetology_0.inputs.x",
                        "mismatching_peers.add_onetology_0.outputs.y",
                        ["http://example.org/NotOutput", "http://example.org/Output"],
                        ["http://example.org/Output"],
                    )
                ],
            )

    def test_derive_from(self):
        out_tag = URIRef(
            f"{clothes_wf.__name__}.{machine_wash.__name__}_0.outputs.clothes"
        )
        inp_tag = URIRef(
            f"{clothes_wf.__name__}.{machine_wash.__name__}_0.inputs.clothes"
        )

        with self.subTest("Inherit type"):
            graph = get_knowledge_graph(clothes_wf._semantikon_workflow)
            out_types = set(graph.objects(out_tag, RDF.type))
            self.assertIn(
                EX.Garment,
                out_types,
                msg="Should inherit from input",
            )
            self.assertIn(
                EX.SomethingElse,
                out_types,
                msg="Should gain from u-specification",
            )

        with self.subTest("Inherit properties"):
            graph = get_knowledge_graph(clothes_wf._semantikon_workflow)
            out_properties = set(graph.objects(out_tag, EX.hasProperty))
            self.assertIn(
                EX.MachineWashable,
                out_properties,
                msg="Should inherit from input",
            )
            self.assertIn(
                EX.Cleaned, out_properties, msg="Should retain from u-specification"
            )

        type_validation_conflict = [
            (out_tag, inp_tag, [EX.SomethingElse, EX.Garment], [EX.Garment])
        ]
        with self.subTest("No other types"):
            graph = get_knowledge_graph(clothes_wf._semantikon_workflow)
            val = validate_values(graph)
            self.assertListEqual(val["missing_triples"], [])
            self.assertEqual(
                val["incompatible_connections"],
                type_validation_conflict,
                msg="Completely unrelated types should be flagged",
            )

        with self.subTest("No type narrowing"):
            g0 = Graph()
            g0.add((EX.SomethingElse, RDFS.subClassOf, EX.Garment))
            graph = get_knowledge_graph(clothes_wf._semantikon_workflow)
            val = validate_values(graph)
            self.assertListEqual(val["missing_triples"], [])
            self.assertListEqual(
                val["incompatible_connections"],
                type_validation_conflict,
                msg="Downstream having a broader type than upstream is disallowed",
            )

        with self.subTest("Type broadening OK"):
            context = Graph()
            context.add((EX.Garment, RDFS.subClassOf, EX.SomethingElse))
            graph = get_knowledge_graph(clothes_wf._semantikon_workflow, graph=context)
            val = validate_values(graph)
            self.assertListEqual(val["missing_triples"], [])
            self.assertListEqual(
                val["incompatible_connections"],
                [],
                msg="Downstream having a narrower type than upstream is fine",
            )

    def test_uri_restrictions_derived_from_interaction(self):
        with self.subTest("Single step"):
            graph = get_knowledge_graph(single._semantikon_workflow)
            val = validate_values(graph)
            self.assertFalse(
                val["missing_triples"] or val["incompatible_connections"],
                msg=f"URI specification, restrictions, and derived_from should all play"
                f"well together. Expected no validation problems, but got {val}",
            )

        with self.subTest("Chain steps"):
            graph = get_knowledge_graph(chain._semantikon_workflow)
            val = validate_values(graph)
            self.assertFalse(
                val["missing_triples"] or val["incompatible_connections"],
                msg=f"URI specification, restrictions, and derived_from should all play"
                f"well together. Expected no validation problems, but got {val}",
            )

    def test_macro(self):
        graph = get_knowledge_graph(get_macro.run())
        subj = list(
            graph.subjects(
                SNS.hasValue,
                URIRef("get_macro.add_three_0.add_one_0.inputs.a.value"),
            )
        )
        self.assertEqual(len(subj), 3)
        subj = list(
            graph.subjects(
                SNS.hasValue,
                URIRef("get_macro.add_three_0.add_two_0.outputs.result.value"),
            )
        )
        self.assertEqual(len(subj), 3)
        for ii, tag in enumerate(
            [
                "add_three_0.add_two_0.outputs.result",
                "add_three_0.outputs.w",
                "add_one_0.inputs.a",
            ]
        ):
            with self.subTest(i=ii):
                self.assertIn(
                    URIRef("get_macro." + tag), subj, msg=f"{tag} not in {subj}"
                )
        inherits_from = [
            (str(g[0]), str(g[1])) for g in graph.subject_objects(PROV.wasDerivedFrom)
        ]
        get_macro_io_passing = 2
        get_three_io_passing = 2
        get_three_internal = 1
        get_macro_internal = 1
        expected_inheritance = (
            get_macro_io_passing
            + get_three_io_passing
            + get_three_internal
            + get_macro_internal
        )
        prefix = "get_macro.add_three_0"
        sub_obj = [
            ("add_one_0.inputs.a", "inputs.c"),
            ("outputs.w", "add_two_0.outputs.result"),
        ]
        sub_obj = [(prefix + "." + s, prefix + "." + o) for s, o in sub_obj]
        self.assertEqual(len(inherits_from), expected_inheritance)
        for ii, pair in enumerate(sub_obj):
            with self.subTest(i=ii):
                self.assertIn(pair, inherits_from)

    def test_macro_full_comparison(self):
        txt = dedent(
            """\
        @prefix ns1: <http://pyiron.org/ontology/> .
        @prefix prov: <http://www.w3.org/ns/prov#> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        
        <get_macro.add_one_0.inputs.a> a prov:Entity ;
            ns1:hasValue <get_macro.add_three_0.add_two_0.outputs.result.value> ;
            prov:wasDerivedFrom <get_macro.add_three_0.outputs.w> ;
            ns1:inputOf <get_macro.add_one_0> .
        
        <get_macro.add_three_0.add_one_0.inputs.a> a prov:Entity ;
            ns1:hasValue <get_macro.add_three_0.add_one_0.inputs.a.value> ;
            prov:wasDerivedFrom <get_macro.add_three_0.inputs.c> ;
            ns1:inputOf <get_macro.add_three_0.add_one_0> .
        
        <get_macro.add_three_0.add_two_0.inputs.b> a prov:Entity ;
            ns1:hasValue <get_macro.add_three_0.add_one_0.outputs.result.value> ;
            prov:wasDerivedFrom <get_macro.add_three_0.add_one_0.outputs.result> ;
            ns1:inputOf <get_macro.add_three_0.add_two_0> .
        
        <get_macro.outputs.result> a prov:Entity ;
            ns1:hasValue <get_macro.add_one_0.outputs.result.value> ;
            prov:wasDerivedFrom <get_macro.add_one_0.outputs.result> ;
            ns1:outputOf <get_macro> .
        
        <get_macro.add_one_0.outputs.result> a prov:Entity ;
            ns1:hasValue <get_macro.add_one_0.outputs.result.value> ;
            ns1:outputOf <get_macro.add_one_0> ;
            ns1:fulfills <get_macro.outputs.result> .
        
        <get_macro.add_three_0.add_one_0.outputs.result> a prov:Entity ;
            ns1:hasValue <get_macro.add_three_0.add_one_0.outputs.result.value> ;
            ns1:outputOf <get_macro.add_three_0.add_one_0> ;
            ns1:fulfills <get_macro.add_three_0.add_two_0.inputs.b> .
        
        <get_macro.add_three_0.add_two_0.outputs.result> a prov:Entity ;
            ns1:hasValue <get_macro.add_three_0.add_two_0.outputs.result.value> ;
            ns1:outputOf <get_macro.add_three_0.add_two_0> ;
            ns1:fulfills <get_macro.add_three_0.outputs.w> .
        
        <get_macro.add_three_0.inputs.c> a prov:Entity ;
            ns1:hasValue <get_macro.add_three_0.add_one_0.inputs.a.value> ;
            prov:wasDerivedFrom <get_macro.inputs.c> ;
            ns1:inputOf <get_macro.add_three_0> ;
            ns1:fulfills <get_macro.add_three_0.add_one_0.inputs.a> .
        
        <get_macro.add_three_0.outputs.w> a prov:Entity ;
            ns1:hasValue <get_macro.add_three_0.add_two_0.outputs.result.value> ;
            prov:wasDerivedFrom <get_macro.add_three_0.add_two_0.outputs.result> ;
            ns1:outputOf <get_macro.add_three_0> ;
            ns1:fulfills <get_macro.add_one_0.inputs.a> .
        
        <get_macro.inputs.c> a prov:Entity ;
            ns1:hasValue <get_macro.add_three_0.add_one_0.inputs.a.value> ;
            ns1:inputOf <get_macro> ;
            ns1:fulfills <get_macro.add_three_0.inputs.c> .
        
        <get_macro.add_one_0.outputs.result.value> rdf:value 5 .
        
        <get_macro.add_three_0.add_one_0.outputs.result.value> rdf:value 2 .
        
        <get_macro.add_three_0.add_one_0.inputs.a.value> rdf:value 1 .
        
        <get_macro.add_three_0.add_two_0.outputs.result.value> rdf:value 4 .
        
        <get_macro> a prov:Activity ;
            ns1:hasNode <get_macro.add_one_0>,
                <get_macro.add_three_0> .
        
        <get_macro.add_one_0> a prov:Activity ;
            ns1:hasSourceFunction <add_one> .
        
        <get_macro.add_three_0.add_one_0> a prov:Activity ;
            ns1:hasSourceFunction <add_one> .
        
        <get_macro.add_three_0> a prov:Activity ;
            ns1:hasNode <get_macro.add_three_0.add_one_0>,
                <get_macro.add_three_0.add_two_0> .
        
        <get_macro.add_three_0.add_two_0> a prov:Activity ;
            ns1:hasSourceFunction <add_two> .\n\n"""
        )
        ref_graph = Graph()
        ref_graph.parse(data=txt, format="turtle", publicID="")
        original_graph = get_knowledge_graph(get_macro.run())
        graph_to_compare = Graph()
        graph_to_compare = graph_to_compare.parse(data=original_graph.serialize())
        _, in_first, in_second = graph_diff(graph_to_compare, ref_graph)
        self.assertEqual(
            len(in_second), 0, msg=f"Missing triples: {in_second.serialize()}"
        )
        self.assertEqual(
            len(in_first), 0, msg=f"Unexpected triples: {in_first.serialize()}"
        )

    def test_parse_cancel(self):
        channels, edges = serialize_data(get_wrong_order._semantikon_workflow)[1:]
        self.assertTrue(
            any(
                "cancel" in channel["extra"]
                for channel in channels.values()
                if "extra" in channel
            )
        )
        to_cancel = _parse_cancel(channels)
        self.assertEqual(len(to_cancel), 1)
        self.assertEqual(
            to_cancel[0],
            (
                URIRef("get_wrong_order.create_vacancy_0.outputs.structure"),
                URIRef("http://example.org/hasState"),
                URIRef("http://example.org/relaxed"),
            ),
        )

    def test_wrong_order(self):
        graph = get_knowledge_graph(get_wrong_order._semantikon_workflow)
        missing_triples = [
            [str(gg) for gg in g] for g in validate_values(graph)["missing_triples"]
        ]
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

    def test_cancel_and_fulfills(self):
        with self.subTest("Early cancellation"):
            graph = get_knowledge_graph(
                get_right_order_for_fulfillment._semantikon_workflow
            )
            self.assertFalse(
                validate_values(graph)["broken_promises"],
                msg="If the cancel comes before there is anything to cancel, it should "
                "be harmless",
            )

        with self.subTest("Late cancellation"):
            graph = get_knowledge_graph(
                get_wrong_order_for_fulfillment._semantikon_workflow
            )
            o_port = URIRef(
                "get_wrong_order_for_fulfillment.create_vacancy_0.outputs.structure"
            )
            i_port = URIRef(
                "get_wrong_order_for_fulfillment."
                "get_vacancy_formation_energy_fulfills_0.inputs.structure"
            )
            cancelled = (EX.hasState, EX.relaxed)
            self.assertDictEqual(
                {o_port: (i_port, {cancelled})},
                validate_values(graph)["broken_promises"],
                msg="If the cancel comes after a triple has been added, it should be "
                "removed -- here it was necessary.",
            )

    def test_serialize_data(self):
        data = get_macro.run()
        nodes, channels, edges = serialize_data(data)
        for key, node in channels.items():
            self.assertTrue(key.startswith(node[NS.PREFIX]))
            self.assertIn(node[NS.PREFIX], nodes)
        self.assertIn("get_macro.add_three_0.inputs.c", channels)
        for args in edges:
            self.assertIn(args[0], channels)
            self.assertIn(args[1], channels)

    def test_visualize(self):
        data = get_macro.run()
        graph = get_knowledge_graph(data)
        self.assertIsInstance(visualize(graph), Digraph)

    def test_function_referencing(self):
        graph = get_knowledge_graph(get_correct_analysis_owl._semantikon_workflow)
        self.assertEqual(
            list(graph.subject_objects(PROV.wasGeneratedBy))[0],
            (
                URIRef("get_correct_analysis_owl.add_0.inputs.a"),
                URIRef("get_correct_analysis_owl.add_0"),
            ),
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


@workflow
def get_run_md(inp: Input):
    result = run_md(inp)
    return result


class Animal:
    class Mammal:
        class Dog:
            pass

        class Cat:
            pass

    class Reptile:
        class Lizard:
            pass

        class Snake:
            pass


class ForbiddenAnimal:
    class Mamal:
        class ForbiddenAnimal:
            pass


class TestDataclass(unittest.TestCase):
    def test_dataclass(self):
        wf_dict = get_run_md.run(Input(T=300.0, n=100))
        graph = get_knowledge_graph(wf_dict)
        i_txt = "get_run_md.run_md_0.inputs.inp"
        o_txt = "get_run_md.run_md_0.outputs.out"
        triples = (
            (URIRef(f"{i_txt}.n.value"), RDFS.subClassOf, URIRef(f"{i_txt}.value")),
            (URIRef(f"{i_txt}.n.value"), RDF.value, Literal(100)),
            (URIRef(f"{i_txt}.parameters.a.value"), RDF.value, Literal(2)),
            (URIRef(o_txt), SNS.hasValue, URIRef(f"{o_txt}.E.value")),
        )
        for ii, triple in enumerate(triples):
            with self.subTest(i=ii):
                self.assertEqual(
                    len(list(graph.triples(triple))),
                    1,
                    msg=f"{triple} not found",
                )
        self.assertIsNone(graph.value(URIRef(f"{i_txt}.not_dataclass.b.value")))

    def test_dataclass_to_knowledge_graph(self):
        EX = Namespace("http://example.org/")
        tags = [
            ("Cat", "Mammal"),
            ("Lizard", "Reptile"),
            ("Mammal", "Animal"),
            ("Dog", "Mammal"),
            ("Snake", "Reptile"),
            ("Reptile", "Animal"),
        ]
        doubles = sorted([(EX[tag[0]], EX[tag[1]]) for tag in tags])
        graph = dataclass_to_knowledge_graph(Animal, EX)
        self.maxDiff = None
        self.assertEqual(sorted(graph.subject_objects(None)), doubles)
        with self.assertRaises(ValueError):
            graph = dataclass_to_knowledge_graph(ForbiddenAnimal, EX)


if __name__ == "__main__":
    unittest.main()
