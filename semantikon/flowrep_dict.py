"""Helpers for converting callables and annotations to flowrep metadata."""

from __future__ import annotations

from typing import Annotated, Any, get_args, get_origin

import flowrep as fr

from semantikon.converter import parse_metadata
from semantikon.datastructure import TypeMetadata


def _flowrep_recipe_from_callable(function: Any, *, node_type: str):
    if hasattr(function, "flowrep_recipe"):
        return function.flowrep_recipe
    decorator = fr.tools.atomic if node_type == "atomic" else fr.tools.workflow
    return decorator(function).flowrep_recipe


def annotation_to_type_hint(annotation: Any) -> Any | None:
    """
    Extract the underlying type hint from a parameter annotation.

    Plain annotations (``int`` in ``def foo(x: int)``) are returned as-is;
    ``Annotated[T, ...]`` is unwrapped to ``T``.

    Args:
        annotation: A parameter annotation, plain or ``Annotated``.

    Returns:
        The underlying type, or ``None`` if ``annotation`` is ``None``.
    """
    if annotation is None:
        return None
    elif get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    else:
        return annotation


def annotation_to_type_metadata(annotation: Any) -> TypeMetadata | None:
    """
    Extract ``TypeMetadata`` from an ``Annotated`` parameter annotation.

    Plain annotations (``int`` in ``def foo(x: int)``) carry no metadata and
    yield ``None``; only ``Annotated[T, ...]`` is parsed.

    Args:
        annotation: A parameter annotation, plain or ``Annotated``.

    Returns:
        The parsed metadata, or ``None`` if ``annotation`` is not ``Annotated``.
    """
    if annotation is None or get_origin(annotation) is not Annotated:
        return None
    return parse_metadata(annotation)
