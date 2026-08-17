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


def _to_node(node_str: str) -> Node | Input | Output:
    """Convert a compound node name to the appropriate dataclass key.

    Args:
        node_str (str): A node name string, e.g. ``"prefix-inputs-port"`` or
            ``"prefix-outputs-port"`` or ``"prefix-step"``.

    Returns:
        Node | Input | Output: The corresponding dataclass instance.
    """
    if "-inputs-" in node_str:
        node_part, port_part = node_str.split("-inputs-", 1)
        return Input(node=node_part, port=port_part)
    elif "-outputs-" in node_str:
        node_part, port_part = node_str.split("-outputs-", 1)
        return Output(node=node_part, port=port_part)
    return Node(name=node_str)


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

    for position, inp in enumerate(wf.inputs):
        inp_node = Input(node=prefix, port=_get_name(inp.id))
        inp_position = position
        if inp.inputBinding is not None and inp.inputBinding.position is not None:
            inp_position = inp.inputBinding.position
        G.add_node(inp_node, step="inputs", position=inp_position)
    for position, out in enumerate(wf.outputs):
        out_node = Output(node=prefix, port=_get_name(out.id))
        G.add_node(out_node, step="outputs", position=position)

    if isinstance(wf, parser.CommandLineTool):
        return G

    for step in wf.steps:
        node_name = f"{prefix}-{_get_name(step.id)}"
        run_doc = parser.load_document_by_uri(step.run)
        node_type = "workflow" if isinstance(run_doc, parser.Workflow) else "atomic"
        G.add_node(Node(name=node_name), type=node_type, step="node")
        for inp in step.in_:
            source = _get_name(inp.source)
            source = (
                source.replace("/", "-outputs-")
                if "/" in source
                else f"inputs-{source}"
            )
            dest = f"{prefix}-{_get_name(inp.id).replace('/', '-inputs-')}"
            G.add_edge(_to_node(f"{prefix}-{source}"), _to_node(dest))
            G.add_edge(_to_node(dest), Node(name=node_name))
        for out in step.out:
            out_name = _get_name(out)
            if "/" in out_name:
                out_name = out_name.split("/")[-1]
            G.add_edge(Node(name=node_name), Output(node=node_name, port=out_name))
        G = _add_node(run_doc, G, prefix=node_name)
    for out in wf.outputs:
        G.add_edge(
            _to_node(
                f"{prefix}-{_get_name(out.outputSource.replace('/', '-outputs-'))}"
            ),
            Output(node=prefix, port=_get_name(out.id)),
        )
    return G
