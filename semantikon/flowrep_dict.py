"""
Convert live node trees to the legacy nested-dictionary format.

This module bridges :mod:`flowrep.models.live` objects (recipe + instance data)
to the ``dict`` structure historically produced by :func:`flowrep.workflow.get_workflow_dict`
and consumed by downstream packages such as *semantikon*.

Edge string format
------------------
``"inputs.{port}"``
    Workflow-level input port.
``"outputs.{port}"``
    Workflow-level output port.
``"{child_label}.inputs.{port}"``
    Child node input port.
``"{child_label}.outputs.{port}"``
    Child node output port.

.. note::

   Port names are taken directly from the flowrep recipe (e.g. ``"output_0"``,
   ``"quotient"``).  The legacy ``workflow.py`` used ``"output"`` for
   single-output atomics and numeric indices for multi-output; the new naming
   is self-consistent within edges and should be transparent to any consumer
   that follows edges rather than hard-coding port names.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, get_args, get_origin

from pyiron_snippets import retrieve

from flowrep import tools
from flowrep.models import edge_models, live
from flowrep.models.nodes import workflow_model


def live_to_dict(
    node: live.LiveNode,
    *,
    with_io: bool = False,
    with_function: bool = False,
    label: str | None = None,
) -> dict[str, Any]:
    """Convert a live node to the nested dictionary format.

    Args:
        node: The live node to convert (pre- or post-run).
        with_io: Include ``"inputs"`` / ``"outputs"`` port dictionaries.
        with_function: Store raw callables (``True``) or
            :func:`tools.get_function_metadata` dicts (``False``).
        label: Override the inferred label.

    Returns:
        A nested dictionary matching the structure of the legacy
        ``get_workflow_dict`` output.
    """
    if isinstance(node, live.Atomic):
        return _atomic_to_dict(node, with_io=with_io, with_function=with_function)
    if isinstance(node, live.Workflow):
        return _workflow_to_dict(
            node, with_io=with_io, with_function=with_function, label=label
        )
    if isinstance(node, live.FlowControl):
        raise NotImplementedError(
            "FlowControl → dict conversion is not yet implemented.  "
            "The legacy format flattens body-workflow children into the "
            "control-flow dict, which requires unwrapping the body recipe."
        )
    raise TypeError(f"Unsupported live node type: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Atomic
# ---------------------------------------------------------------------------


def _atomic_to_dict(
    node: live.Atomic,
    *,
    with_io: bool,
    with_function: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "atomic"}
    result["function"] = (
        node.function if with_function else tools.get_function_metadata(node.function)
    )
    if with_io:
        result["inputs"] = _input_ports_to_dict(node.input_ports)
        result["outputs"] = _output_ports_to_dict(node.output_ports)
    return result


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def _workflow_to_dict(
    node: live.Workflow,
    *,
    with_io: bool,
    with_function: bool,
    label: str | None,
) -> dict[str, Any]:
    recipe = node.recipe
    assert isinstance(recipe, workflow_model.WorkflowNode)

    result: dict[str, Any] = {
        "type": "workflow",
        "label": label or _infer_label(recipe),
        "nodes": {
            child_label: live_to_dict(
                child_node,
                with_io=with_io,
                with_function=with_function,
                label=child_label,
            )
            for child_label, child_node in node.nodes.items()
        },
        "edges": _workflow_edges(recipe),
    }

    if with_io:
        result["inputs"] = _input_ports_to_dict(node.input_ports)
        result["outputs"] = _output_ports_to_dict(node.output_ports)

    if with_function and recipe.reference is not None:
        result["function"] = retrieve.import_from_string(
            recipe.reference.info.fully_qualified_name
        )

    return result


def _infer_label(recipe: workflow_model.WorkflowNode) -> str:
    """Best-effort label from a workflow recipe's reference."""
    if recipe.reference is not None:
        return recipe.reference.info.fully_qualified_name.rsplit(".", 1)[-1]
    return ""


# ---------------------------------------------------------------------------
# Edge conversion
# ---------------------------------------------------------------------------


def _workflow_edges(recipe: workflow_model.WorkflowNode) -> list[tuple[str, str]]:
    """Flatten typed edge objects into ``("src", "tgt")`` string tuples."""
    edges: list[tuple[str, str]] = []

    needs_output_sanitization = {
        # semantikon uses "output" instead of "outputs_0" for default of one return
        label
        for label, child in recipe.nodes.items()
        if len(child.outputs) == 1 and child.outputs[0] == "output_0"
    }

    # Workflow input → child input
    for target, source in recipe.input_edges.items():
        edges.append(
            (
                f"inputs.{source.port}",
                f"{target.node}.inputs.{target.port}",
            )
        )

    # Sibling output → sibling input
    for target, source in recipe.edges.items():
        edges.append(
            (
                f"{source.node}.outputs.{'output' if source.node in needs_output_sanitization else source.port}",
                f"{target.node}.inputs.{target.port}",
            )
        )

    # Child output → workflow output  (or passthrough from workflow input)
    for target, source in recipe.output_edges.items():
        if isinstance(source, edge_models.InputSource):
            edges.append(
                (
                    f"inputs.{source.port}",
                    f"outputs.{'output' if target.node in needs_output_sanitization else target.port}",
                )
            )
        else:
            edges.append(
                (
                    f"{source.node}.outputs.{'output' if source.node in needs_output_sanitization else source.port}",
                    f"outputs.{target.port}",
                )
            )

    return edges


# ---------------------------------------------------------------------------
# Port conversion
# ---------------------------------------------------------------------------


def _port_dict(value, annotation, default=live.NOT_DATA):
    d = {}
    if not isinstance(value, live.NotData):
        d["value"] = value
    if annotation is not None:
        d["dtype"] = _unwrap_annotated(annotation)
    if not isinstance(default, live.NotData):
        d["default"] = default
    return d


def _unwrap_annotated(annotation: Any) -> Any:
    """
    Strip ``Annotated`` wrappers, returning just the base type.

    Semantic metadata (``uri``, ``units``, ``triples``, …) embedded in
    ``Annotated`` extras belongs to *semantikon*, not to the flowrep dict.
    Storing the full ``Annotated`` type as ``"dtype"`` would cause
    ``meta_to_dict`` to double-extract that metadata in a wrong context.
    """
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def _input_ports_to_dict(
    ports: Mapping[str, live.InputPort],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, port in ports.items():
        result[name] = _port_dict(port.value, port.annotation, port.default)
    return result


def _output_ports_to_dict(
    ports: Mapping[str, live.OutputPort],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if len(ports) == 1 and next(iter(ports.keys())) == "output_0":
        port = next(iter(ports.values()))
        result["output"] = _port_dict(port.value, port.annotation)
    else:
        for name, port in ports.items():
            result[name] = _port_dict(port.value, port.annotation)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
