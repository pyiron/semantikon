from __future__ import annotations

from pathlib import Path
from typing import Any, cast

try:
    from cwl_utils import parser
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "semantikon.cwl requires optional CWL dependencies. Install with `pip install semantikon[cwl]`."
    ) from exc

from rdflib import RDF, Graph, Literal, URIRef
from schema_salad.utils import yaml_no_ts

from semantikon import ontology
from semantikon.flowrep_to_networkx import Input, Node, Output


def serialize_and_convert_to_networkx(uri: str | Path) -> ontology.SemantikonDiGraph:
    """
    Parse a CWL document and build a knowledge graph.

    Args:
        uri (str | Path): Path or URI to the CWL file.

    Returns:
        ontology.SemantikonDiGraph: A directed graph representing the workflow
            structure, with nodes for inputs, outputs, and steps, and edges
            representing data flow between them.
    """
    wf = parser.load_document_by_uri(uri)
    return _add_node(wf)


def _get_name(tag: str) -> str:
    """
    Extract the local name from a CWL identifier URI.

    CWL identifiers are typically full URIs or fragment identifiers of the form
    ``file:///path/to/file.cwl#local_name``. This function returns the part after
    the ``#`` character, or the full string if no ``#`` is present.

    Args:
        tag (str): A CWL identifier string.

    Returns:
        str: The local name extracted from the identifier.
    """
    return tag.split("#")[-1]


def _add_node(
    wf: parser.CommandLineTool | parser.Workflow,
    G: ontology.SemantikonDiGraph | None = None,
    prefix: Node | None = None,
) -> ontology.SemantikonDiGraph:
    """
    Recursively add nodes and edges for a CWL process to the knowledge graph.

    For a ``CommandLineTool``, input and output nodes are added. For a
    ``Workflow``, step nodes are also added along with edges representing the
    data flow between steps.

    Args:
        wf (parser.CommandLineTool | parser.Workflow): The CWL process to add
            to the graph.
        G (ontology.SemantikonDiGraph | None): The graph to populate. If
            ``None``, a new graph is created using the workflow's filename as
            the prefix.
        prefix (str | None): The node name prefix. If ``None``, derived from
            the CWL filename (without the ``.cwl`` extension).

    Returns:
        ontology.SemantikonDiGraph: The populated knowledge graph.
    """
    if prefix is None:
        prefix = Node(name=wf.id.split("/")[-1].replace(".cwl", ""))
    if G is None:
        G = ontology.SemantikonDiGraph(prefix=str(prefix))

    for position, inp in enumerate(wf.inputs):
        inp_node = Input(node=prefix, port=_get_name(inp.id))
        inp_position = position
        if inp.inputBinding is not None and inp.inputBinding.position is not None:
            inp_position = inp.inputBinding.position
        G.add_node(inp_node, position=inp_position)
        G.add_edge(inp_node, prefix)

    for position, out in enumerate(wf.outputs):
        out_node = Output(node=prefix, port=_get_name(out.id))
        G.add_node(out_node, position=position)
        G.add_edge(prefix, out_node)

    if isinstance(wf, parser.CommandLineTool):
        return G

    for step in wf.steps:
        node = Node(owner=prefix, name=_get_name(step.id))
        run_doc = parser.load_document_by_uri(step.run)
        node_type = "workflow" if isinstance(run_doc, parser.Workflow) else "atomic"
        G.add_node(node, type=node_type)
        for inp in step.in_:
            n, p = _get_name(inp.id).split("/")
            dest = Input(node=Node(owner=prefix, name=n), port=p)
            s = _get_name(inp.source)
            if "/" in s:
                n, p = s.split("/")
                G.add_edge(Output(node=Node(owner=prefix, name=n), port=p), dest)
            else:
                G.add_edge(Input(node=prefix, port=s), dest)
            G.add_edge(dest, node)
        for out in step.out:
            out_name = _get_name(out)
            if "/" in out_name:
                n, p = out_name.split("/")
                G.add_edge(node, Output(node=Node(owner=prefix, name=n), port=p))
            else:
                G.add_edge(node, Output(node=node, port=out_name))
        G = _add_node(run_doc, G, prefix=node)

    for out in wf.outputs:
        n, p = _get_name(out.outputSource).split("/")
        G.add_edge(
            Output(node=Node(owner=prefix, name=n), port=p),
            Output(node=prefix, port=_get_name(out.id)),
        )
    return G


_CWL_TYPE_BY_PYTHON_TYPE: dict[type, str] = {
    bool: "boolean",
    int: "long",
    float: "double",
    str: "string",
}


def _infer_cwl_type(arg: dict[str, Any]) -> str:
    """
    Infer a CWL type for a function argument.

    ``function_to_knowledge_graph`` does not currently record the Python type
    annotation (``dtype``) of a function's arguments in the knowledge graph,
    so it cannot be recovered when converting back. If a default value is
    available, its Python type is used to pick a corresponding CWL type;
    otherwise the permissive CWL ``Any`` type is used.

    Args:
        arg (dict[str, Any]): Argument metadata as produced by
            ``ontology._graph_to_function``.

    Returns:
        str: The CWL type name.
    """
    if "default" in arg:
        return _CWL_TYPE_BY_PYTHON_TYPE.get(type(arg["default"]), "Any")
    return "Any"


def _arg_to_cwl_input(
    cwl_module: Any, arg: dict[str, Any], position: int
) -> parser.CommandInputParameter:
    """
    Convert function input argument metadata into a CWL input parameter.

    Args:
        cwl_module: The versioned ``cwl_utils.parser`` submodule to build
            objects with (e.g. ``cwl_utils.parser.cwl_v1_2``).
        arg (dict[str, Any]): Argument metadata as produced by
            ``ontology._graph_to_function``.
        position (int): Fallback input/argument position if none is recorded.

    Returns:
        parser.CommandInputParameter: The resulting CWL input parameter.
    """
    kwargs: dict[str, Any] = {
        "id": arg.get("arg", f"input_{position}"),
        "type_": _infer_cwl_type(arg),
        "inputBinding": cwl_module.CommandLineBinding(
            position=arg.get("position", position)
        ),
    }
    if "default" in arg:
        kwargs["default"] = arg["default"]
    return cwl_module.CommandInputParameter(**kwargs)


def _arg_to_cwl_output(
    cwl_module: Any, arg: dict[str, Any], position: int
) -> parser.CommandOutputParameter:
    """
    Convert function output argument metadata into a CWL output parameter.

    Args:
        cwl_module: The versioned ``cwl_utils.parser`` submodule to build
            objects with (e.g. ``cwl_utils.parser.cwl_v1_2``).
        arg (dict[str, Any]): Argument metadata as produced by
            ``ontology._graph_to_function``.
        position (int): Fallback output position if none is recorded.

    Returns:
        parser.CommandOutputParameter: The resulting CWL output parameter.
    """
    return cwl_module.CommandOutputParameter(
        id=arg.get("arg", f"output_{position}"),
        type_=_infer_cwl_type(arg),
    )


def _get_function_id(g: Graph, f_node: URIRef) -> str:
    for denoted_by in g.objects(f_node, ontology.SNS.denoted_by):
        if (denoted_by, RDF.type, ontology.SNS.identifier) in g:
            value = g.value(denoted_by, ontology.SNS.has_value)
            if isinstance(value, Literal):
                return value.toPython()
    raise ValueError(f"Function node {f_node} has no identifier in the graph.")


def knowledge_graph_to_cwl(
    graph: Graph, f_node: URIRef | None = None, cwl_version: str = "v1.2"
) -> parser.CommandLineTool:
    """
    Convert a function stored in a knowledge graph into an in-memory CWL
    ``CommandLineTool`` object.

    The knowledge graph is expected to have been produced (at least in part)
    by ``semantikon.ontology.function_to_knowledge_graph``, i.e. it must
    contain a node of type ``SNS.workflow_function`` describing the function's
    inputs and outputs.

    Args:
        graph (rdflib.Graph): Knowledge graph containing the function
            description.
        f_node (rdflib.URIRef | None): URI of the function node to convert.
            If ``None``, the graph must contain exactly one node of type
            ``SNS.workflow_function``.
        cwl_version (str): CWL schema version to target, e.g. ``"v1.0"``,
            ``"v1.1"`` or ``"v1.2"``.

    Returns:
        parser.CommandLineTool: The resulting CWL tool description.
    """
    if f_node is None:
        candidates = list(graph.subjects(RDF.type, ontology.SNS.workflow_function))
        if len(candidates) != 1:
            raise ValueError(
                "f_node must be provided explicitly unless the graph contains "
                f"exactly one function node (found {len(candidates)})."
            )
        f_node = cast(URIRef, candidates[0])

    data = ontology._graph_to_function(graph, f_node)
    cwl_module = getattr(parser, f"cwl_{cwl_version.replace('.', '_')}")

    inputs = [
        _arg_to_cwl_input(cwl_module, arg, position)
        for position, arg in enumerate(data["input_args"])
    ]
    outputs = [
        _arg_to_cwl_output(cwl_module, arg, position)
        for position, arg in enumerate(data["output_args"])
    ]

    return cwl_module.CommandLineTool(
        id=_get_function_id(graph, f_node).replace(":", "_"),
        inputs=inputs,
        outputs=outputs,
        doc=data["data"].get("docstring") or None,
        cwlVersion=cwl_version,
    )


def save_cwl_file(tool: parser.CommandLineTool, uri: str | Path) -> Path:
    """
    Serialize an in-memory CWL tool (e.g. as built by
    ``knowledge_graph_to_cwl``) to a ``.cwl`` file on disk.

    Args:
        tool (parser.CommandLineTool): The CWL tool object to serialize.
        uri (str | Path): Destination path of the ``.cwl`` file.

    Returns:
        Path: The path the file was written to.
    """
    path = Path(uri)
    saved = parser.save(tool, base_url=path.resolve().as_uri())
    with path.open("w") as f:
        f.write("#!/usr/bin/env cwl-runner\n")
        yaml_no_ts().dump(saved, f)
    return path
