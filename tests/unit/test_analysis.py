import unittest
from dataclasses import dataclass
from typing import Annotated

from rdflib import RDF, Namespace, URIRef

from semantikon import analysis as asis
from semantikon import ontology as onto
from semantikon.metadata import meta
from semantikon.workflow import workflow

EX: Namespace = Namespace("http://example.org/")
PMD: Namespace = Namespace("https://w3id.org/pmd/co/PMD_")


@dataclass
class SpeedData:
    distance: Annotated[
        float, {"uri": PMD["0040001"], "units": "meter", "label": "Distance"}
    ]
    time: Annotated[float, {"units": "second"}]


def get_speed(
    distance: Annotated[
        float, {"uri": PMD["0040001"], "units": "meter", "label": "Distance"}
    ],
    time: Annotated[float, {"units": "second"}],
) -> Annotated[float, {"units": "meter/second", "uri": EX.Velocity, "label": "speed"}]:
    """some random docstring"""
    speed = distance / time
    return speed


def get_speed_with_dataclass(
    data: SpeedData,
) -> Annotated[float, {"units": "meter/second", "uri": EX.Velocity, "label": "speed"}]:
    """some random docstring"""
    speed = data.distance / data.time
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


@workflow
def workflow_with_dataclass(data: SpeedData, mass):
    speed = get_speed_with_dataclass(data)
    kinetic_energy = get_kinetic_energy(mass, speed)
    return kinetic_energy


@workflow
def only_get_speed_workflow(distance, time):
    speed = get_speed(distance=distance, time=time)
    return speed


class TestAnalysis(unittest.TestCase):
    def test_my_kinetic_energy_workflow_graph(self):
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        g = onto.get_knowledge_graph(wf_dict, prefix="T")

        with self.subTest("workflow instance exists"):
            uri = asis.identifier_to_uri(g, "my_kinetic_energy_workflow")[0]
            workflows = list(g.subjects(RDF.type, uri))
            self.assertEqual(len(workflows), 1)

        wf = workflows[0]

        with self.subTest("workflow has both function executions as parts"):
            parts = list(g.objects(wf, onto.BFO["0000051"]))
            uri = asis.identifier_to_uri(g, "get_kinetic_energy_0")[0]
            ke_calls = [p for p in parts if (p, RDF.type, uri) in g]
            uri = asis.identifier_to_uri(g, "get_speed_0")[0]
            speed_calls = [p for p in parts if (p, RDF.type, uri) in g]
            self.assertEqual(len(ke_calls), 1)
            self.assertEqual(len(speed_calls), 1)

        ke_call = ke_calls[0]
        speed_call = speed_calls[0]

        with self.subTest("speed computation precedes kinetic energy computation"):
            self.assertIn((speed_call, onto.BFO["0000063"], ke_call), g)

        with self.subTest("functions are linked to python callables"):
            self.assertIn(
                (
                    ke_call,
                    onto.RO["0000059"],
                    onto.BASE[
                        f"{__name__}-get_kinetic_energy-not_defined".replace(".", "-")
                    ],
                ),
                g,
            )
            self.assertIn(
                (
                    speed_call,
                    onto.RO["0000059"],
                    onto.BASE[f"{__name__}-get_speed-not_defined".replace(".", "-")],
                ),
                g,
            )

    def test_request_values(self):
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        wf_dict["inputs"]["distance"]["value"] = 1.0
        wf_dict["inputs"]["time"]["value"] = 2.0
        wf_dict["inputs"]["mass"]["value"] = 3.0
        self.assertDictEqual(wf_dict["outputs"], {"kinetic_energy": {}})
        graph = onto.get_knowledge_graph(
            my_kinetic_energy_workflow.run(distance=2.0, time=2.0, mass=3.0)
        )
        wf_dict = asis.request_values(wf_dict, graph)
        self.assertDictEqual(
            wf_dict["outputs"], {"kinetic_energy": {}}, msg="no known inputs"
        )
        graph += onto.get_knowledge_graph(
            my_kinetic_energy_workflow.run(distance=1.0, time=2.0, mass=3.0)
        )
        wf_dict = asis.request_values(wf_dict, graph)
        self.assertDictEqual(
            wf_dict["outputs"],
            {"kinetic_energy": {"value": 0.375}},
            msg="all inputs known because the same simulation was run before",
        )
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        wf_dict["inputs"]["distance"]["value"] = 1.0
        wf_dict["inputs"]["time"]["value"] = 2.0
        wf_dict["inputs"]["mass"]["value"] = 4.0
        wf_dict = asis.request_values(wf_dict, graph)
        self.assertDictEqual(
            wf_dict["outputs"],
            {"kinetic_energy": {}},
            msg="kinetic energy must be unknown because of unknown mass",
        )
        self.assertEqual(
            wf_dict["nodes"]["get_speed_0"]["outputs"]["speed"]["value"],
            0.5,
            msg="speed must be known because of known distance and time",
        )
        graph = onto.get_knowledge_graph(
            my_kinetic_energy_workflow.run(distance=1.0, time=2.0, mass=4.0),
            remove_data=True,
        )
        wf_dict = asis.request_values(wf_dict, graph)
        self.assertDictEqual(
            wf_dict["outputs"], {"kinetic_energy": {}}, msg="data not stored"
        )

    def test_sparql_writer(self):
        wf_dict = my_kinetic_energy_workflow.run(distance=2.0, time=1.0, mass=4.0)
        graph = onto.get_knowledge_graph(wf_dict)
        comp = asis.query_io_completer(graph)
        self.assertEqual(
            dir(comp.my_kinetic_energy_workflow.get_speed_0.inputs),
            ["distance", "time"],
        )
        A = comp.my_kinetic_energy_workflow.inputs.time
        self.assertEqual(dir(A), ["query", "to_query_text"])
        B = comp.my_kinetic_energy_workflow.outputs.kinetic_energy
        C = comp.my_kinetic_energy_workflow.inputs.mass
        D = comp.my_kinetic_energy_workflow.inputs.distance
        self.assertListEqual(
            dir(comp.my_kinetic_energy_workflow),
            ["get_kinetic_energy_0", "get_speed_0", "inputs", "outputs"],
        )
        self.assertEqual((A & B).query(), [(1.0, 8.0)])
        self.assertEqual(
            (A & C & B).query(), [(1.0, 4.0, 8.0)], msg=(A & C & B).to_query_text()
        )
        self.assertEqual((A & (C & B)).query(), [(1.0, 4.0, 8.0)])
        self.assertEqual((A & C & D & B).query(), [(1.0, 4.0, 2.0, 8.0)])
        self.assertEqual(((A & C) & (D & B)).query(), [(1.0, 4.0, 2.0, 8.0)])
        self.assertEqual(A.query(), [(1.0,)])
        A_dash = A.value()  # A is now URIRef instead of _Node
        self.assertEqual((A_dash & (C & B)).query(), [(1.0, 4.0, 8.0)])
        self.assertEqual((A_dash & C & D & B).query(), [(1.0, 4.0, 2.0, 8.0)])
        self.assertEqual(((A_dash & C) & (D & B)).query(), [(1.0, 4.0, 2.0, 8.0)])
        self.assertEqual(list(graph.query(A.to_query_text()))[0][0].toPython(), 1.0)
        with self.assertRaises(AttributeError):
            _ = comp.non_existing_node
        self.assertIsInstance(comp.my_kinetic_energy_workflow, asis._Node)

        graph += onto.get_knowledge_graph(
            only_get_speed_workflow.run(distance=3.0, time=1.5), prefix="T"
        )
        comp = asis.query_io_completer(graph)
        A = comp.my_kinetic_energy_workflow.inputs.time
        B = comp.my_kinetic_energy_workflow.outputs.kinetic_energy
        C = comp.my_kinetic_energy_workflow.inputs.mass
        self.assertEqual((A & B).query(), [(1.0, 8.0)])
        self.assertEqual((A & C & B).query(), [(1.0, 4.0, 8.0)])
        self.assertEqual(A.query(), [(1.0,)])
        self.assertListEqual(
            dir(comp), ["my_kinetic_energy_workflow", "only_get_speed_workflow"]
        )
        E = comp.only_get_speed_workflow.inputs.distance
        with self.assertRaises(ValueError) as context:
            _ = (A & E).query()
        self.assertEqual(str(context.exception), "No common head node found")
        self.assertEqual(E.query(), [(3.0,)])
        graph = onto.get_knowledge_graph(wf_dict, remove_data=True)
        comp = asis.query_io_completer(graph)
        A = comp.my_kinetic_energy_workflow.inputs.time
        B = comp.my_kinetic_energy_workflow.outputs.kinetic_energy
        self.assertListEqual((A & B).query(), [])
        data = (A & B).query(fallback_to_hash=True)
        self.assertEqual(data[0][0], 1.0)
        self.assertIsInstance(data[0][1], str)
        self.assertIsInstance(B.query(fallback_to_hash=True)[0][0], str)

    def test_sparql_writer_with_dataclass(self):
        data = SpeedData(distance=1.0, time=2.0)
        wf_dict = workflow_with_dataclass.run(data=data, mass=3.0)
        graph = onto.get_knowledge_graph(wf_dict, extract_dataclasses=True)
        comp = asis.query_io_completer(graph)
        self.assertEqual(
            comp.workflow_with_dataclass.inputs.data.distance.query(),
            [(1.0,)],
        )

    def test_identifier_to_uri(self):
        wf_dict = my_kinetic_energy_workflow.get_semantikon_dict()
        g = onto.get_knowledge_graph(wf_dict)
        uri = asis.identifier_to_uri(g, "my_kinetic_energy_workflow")[0]
        identifier = str(g.value(uri, onto.SNS.local_identifier))
        self.assertEqual(identifier, "my_kinetic_energy_workflow")

    def test_function_request_from_simple_function(self):
        """Test parsing FunctionRequest from a simple function graph."""

        func_uri = URIRef("http://example.org/functions/add")
        func_data = {
            "qualname": "add",
            "docstring": "Add two numbers",
            "module": "math_utils",
        }
        input_args = [
            {"label": "a", "position": 0},
            {"label": "b", "position": 1},
        ]
        output_args = [{"label": "result", "position": 0}]

        graph = onto._function_to_graph(func_uri, func_data, input_args, output_args)
        request = asis.parse_function_request(graph)

        self.assertEqual(request.name, "add")
        self.assertEqual(request.docstring, "Add two numbers")
        self.assertEqual(request.python_import, "math_utils")
        self.assertEqual(request.artifact_type, "function")
        self.assertEqual(len(request.inputs), 2)
        self.assertEqual(len(request.outputs), 1)
        self.assertEqual(request.inputs[0].name, "a")
        self.assertEqual(request.inputs[1].name, "b")
        self.assertEqual(request.outputs[0].name, "result")

    def test_function_request_with_metadata(self):
        """Test FunctionRequest with additional metadata (category, keywords)."""

        func_uri = URIRef("http://example.org/functions/distance")
        func_data = {
            "qualname": "euclidean_distance",
            "docstring": "Calculate Euclidean distance",
            "module": "geometry.vector",
        }
        input_args = [
            {"label": "point1", "position": 0},
            {"label": "point2", "position": 1},
        ]
        output_args = [{"label": "distance", "position": 0}]

        graph = onto._function_to_graph(func_uri, func_data, input_args, output_args)
        request = asis.parse_function_request(
            graph, category="geometry", keywords=["distance", "vector", "math"]
        )

        self.assertEqual(request.category, "geometry")
        self.assertEqual(request.keywords, ["distance", "vector", "math"])
        self.assertEqual(request.name, "euclidean_distance")

    def test_function_request_with_defaults(self):
        """Test FunctionRequest preserves default values."""

        func_uri = URIRef("http://example.org/functions/power")
        func_data = {
            "qualname": "power",
            "docstring": "Raise to power",
            "module": "math_utils",
        }
        input_args = [
            {"label": "base", "position": 0},
            {"label": "exponent", "position": 1, "default": 2},
        ]
        output_args = [{"label": "result", "position": 0}]

        graph = onto._function_to_graph(func_uri, func_data, input_args, output_args)
        request = asis.parse_function_request(graph)

        self.assertEqual(request.inputs[1].name, "exponent")
        self.assertEqual(request.inputs[1].default, 2)

    def test_function_request_with_source_code(self):
        """Test FunctionRequest with source code."""

        func_uri = URIRef("http://example.org/functions/identity")
        func_data = {
            "qualname": "identity",
            "docstring": "Identity function",
            "module": "utils",
        }
        input_args = [{"label": "x", "position": 0}]
        output_args = [{"label": "result", "position": 0}]

        graph = onto._function_to_graph(func_uri, func_data, input_args, output_args)
        source_code = "def identity(x):\n    return x"
        request = asis.parse_function_request(graph, source_code=source_code)

        self.assertEqual(request.source_code, source_code)

    def test_function_request_annotation_ordering(self):
        """Test that annotations are ordered by position."""

        func_uri = URIRef("http://example.org/functions/multi_arg")
        func_data = {
            "qualname": "process",
            "docstring": "Process multiple args",
            "module": "utils",
        }
        input_args = [
            {"label": "first", "position": 0},
            {"label": "second", "position": 1},
            {"label": "third", "position": 2},
        ]
        output_args = [{"label": "result", "position": 0}]

        graph = onto._function_to_graph(func_uri, func_data, input_args, output_args)
        request = asis.parse_function_request(graph)

        self.assertEqual(len(request.inputs), 3)
        self.assertEqual(request.inputs[0].name, "first")
        self.assertEqual(request.inputs[1].name, "second")
        self.assertEqual(request.inputs[2].name, "third")

    def test_annotation_fields(self):
        """Test Annotation dataclass fields."""
        from semantikon.analysis import Annotation

        annotation = Annotation(
            name="param1",
            type_="int",
            label="Parameter 1",
            position=0,
            default=42,
            uri="http://example.org/types/int",
        )

        self.assertEqual(annotation.name, "param1")
        self.assertEqual(annotation.type_, "int")
        self.assertEqual(annotation.label, "Parameter 1")
        self.assertEqual(annotation.position, 0)
        self.assertEqual(annotation.default, 42)
        self.assertEqual(annotation.uri, "http://example.org/types/int")

    def test_function_request_pydantic_validation(self):
        """Test FunctionRequest Pydantic validation."""
        from semantikon.analysis import Annotation, FunctionRequest

        # Minimal valid request
        req = FunctionRequest(name="test_func")
        self.assertEqual(req.name, "test_func")
        self.assertEqual(req.author_name, "unknown")
        self.assertEqual(req.author_email, "unknown")
        self.assertEqual(req.artifact_type, "function")

        # Full request
        req_full = FunctionRequest(
            author_name="John Doe",
            author_email="john@example.com",
            name="complex_func",
            category="data",
            keywords=["transform", "filter"],
            homepage_url="https://example.com",
            documentation_url="https://docs.example.com",
            source_url="https://github.com/example",
            python_import="my_package.module",
            dependencies=["numpy", "pandas"],
            source_code="# code",
            docstring="Function description",
            inputs=[Annotation(name="data")],
            outputs=[Annotation(name="result")],
        )

        self.assertEqual(req_full.author_name, "John Doe")
        self.assertEqual(req_full.author_email, "john@example.com")
        self.assertEqual(len(req_full.keywords), 2)
        self.assertEqual(len(req_full.dependencies), 2)

    def test_function_request_model_dump(self):
        """Test FunctionRequest model_dump method."""

        func_uri = URIRef("http://example.org/functions/test")
        func_data = {
            "qualname": "test",
            "docstring": "Test function",
            "module": "test_module",
        }
        input_args = [{"label": "x", "position": 0}]
        output_args = [{"label": "y", "position": 0}]

        graph = onto._function_to_graph(func_uri, func_data, input_args, output_args)
        request = asis.parse_function_request(graph, category="test")

        data = request.model_dump()

        self.assertIsInstance(data, dict)
        self.assertEqual(data["name"], "test")
        self.assertEqual(data["python_import"], "test_module")
        self.assertEqual(data["category"], "test")
        self.assertEqual(len(data["inputs"]), 1)
        self.assertEqual(len(data["outputs"]), 1)

    def test_extract_annotations_from_graph(self):
        """Test extract_annotations_from_graph helper function."""
        from rdflib import URIRef

        from semantikon.analysis import extract_annotations_from_graph

        func_uri = URIRef("http://example.org/functions/multi")
        func_data = {
            "qualname": "multi",
            "docstring": "Multi arg function",
            "module": "utils",
        }
        input_args = [
            {"label": "a", "position": 0},
            {"label": "b", "position": 1},
        ]
        output_args = [{"label": "result", "position": 0}]

        graph = onto._function_to_graph(func_uri, func_data, input_args, output_args)

        inputs = extract_annotations_from_graph(graph, arg_type="input")
        outputs = extract_annotations_from_graph(graph, arg_type="output")

        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(inputs[0].name, "a")
        self.assertEqual(inputs[1].name, "b")
        self.assertEqual(outputs[0].name, "result")


if __name__ == "__main__":
    unittest.main()
