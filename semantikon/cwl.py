from __future__ import annotations

from pathlib import Path

try:
    from cwl_utils import parser
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "semantikon.cwl requires optional CWL dependencies. Install with `pip install semantikon[cwl]`."
    ) from exc

from semantikon import ontology


def get_knowledge_graph(uri: str | Path) -> ontology.SemantikonDiGraph:
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
    prefix: str | None = None,
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
        prefix = wf.id.split("/")[-1].replace(".cwl", "")
    if G is None:
        G = ontology.SemantikonDiGraph(prefix=prefix)

    for inp in wf.inputs:
        metadata: dict[str, str | int] = {"step": "inputs"}
        if inp.inputBinding is not None:
            metadata["position"] = inp.inputBinding.position
        G.add_node(f"{prefix}-inputs-{_get_name(inp.id)}", **metadata)
    for out in wf.outputs:
        metadata = {"step": "outputs"}
        G.add_node(f"{prefix}-outputs-{_get_name(out.id)}", **metadata)

    if isinstance(wf, parser.CommandLineTool):
        return G

    for step in wf.steps:
        node_name = f"{prefix}-{_get_name(step.id)}"
        G.add_node(node_name, step="node")
        for inp in step.in_:
            source = _get_name(inp.source)
            source = (
                source.replace("/", "-outputs-")
                if "/" in source
                else f"inputs-{source}"
            )
            dest = f"{prefix}-{_get_name(inp.id).replace('/', '-inputs-')}"
            G.add_edge(f"{prefix}-{source}", dest)
            G.add_edge(dest, node_name)
        for out in step.out:
            out_name = _get_name(out)
            if "/" in out_name:
                out_name = out_name.split("/")[-1]
            G.add_edge(node_name, f"{node_name}-outputs-{out_name}")
        G = _add_node(parser.load_document_by_uri(step.run), G, prefix=node_name)
    for out in wf.outputs:
        G.add_edge(
            f"{prefix}-{_get_name(out.outputSource.replace('/', '-outputs-'))}",
            f"{prefix}-outputs-{_get_name(out.id)}",
        )
    return G
