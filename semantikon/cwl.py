from cwl_utils import parser
from semantikon import ontology


def get_knowledge_graph(uri):
    wf = parser.load_document_by_uri(uri)
    return _add_node(wf)


def _get_name(tag):
    return tag.split("#")[-1]


def _add_node(wf, G=None, prefix=None):
    if prefix is None:
        prefix = wf.id.split("/")[-1].replace(".cwl", "")
    if G is None:
        G = ontology.SemantikonDiGraph(prefix=prefix)

    for inp in wf.inputs:
        metadata = {"step": "inputs"}
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
            source = source.replace('/', '-outputs-') if "/" in source else f"inputs-{source}"
            dest = f"{prefix}-{_get_name(inp.id).replace('/', '-inputs-')}"
            G.add_edge(f"{prefix}-{source}", dest)
            G.add_edge(dest, node_name)
        for out in step.out:
            G.add_edge(
                node_name,
                f"{prefix}-{_get_name(out).replace('/', '-outputs-')}"
            )
        G = _add_node(parser.load_document_by_uri(step.run), G, prefix=node_name)
    for out in wf.outputs:
        G.add_edge(
            f"{prefix}-{_get_name(out.outputSource.replace('/', '-outputs-'))}",
            f"{prefix}-outputs-{_get_name(out.id)}"
        )
    return G
