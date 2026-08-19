from __future__ import annotations

from pathlib import Path

try:
    from cwl_utils import parser
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "semantikon.cwl requires optional CWL dependencies. Install with `pip install semantikon[cwl]`."
    ) from exc

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

    for position, out in enumerate(wf.outputs):
        out_node = Output(node=prefix, port=_get_name(out.id))
        G.add_node(out_node, position=position)

    if isinstance(wf, parser.CommandLineTool):
        return G

    for step in wf.steps:
        node = Node(parent=prefix, name=_get_name(step.id))
        run_doc = parser.load_document_by_uri(step.run)
        node_type = "workflow" if isinstance(run_doc, parser.Workflow) else "atomic"
        G.add_node(node, type=node_type)
        for inp in step.in_:
            n, p = _get_name(inp.id).split("/")
            dest = Input(node=Node(parent=prefix, name=n), port=p)
            s = _get_name(inp.source)
            if "/" in s:
                n, p = s.split("/")
                G.add_edge(Output(node=Node(parent=prefix, name=n), port=p), dest)
            else:
                G.add_edge(Input(node=prefix, port=s), dest)
            G.add_edge(dest, node)
        for out in step.out:
            out_name = _get_name(out)
            if "/" in out_name:
                n, p = out_name.split("/")
                G.add_edge(node, Output(node=Node(parent=prefix, name=n), port=p))
            else:
                G.add_edge(node, Output(node=node, port=out_name))
        G = _add_node(run_doc, G, prefix=node)
    for out in wf.outputs:
        node = Node(parent=prefix, name=_get_name(out.id))
        n, p = _get_name(out.outputSource).split("/")
        G.add_edge(
            Output(node=Node(parent=prefix, name=n), port=p),
            Output(node=prefix, port=_get_name(out.id)),
        )
    return G
