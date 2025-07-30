from typing import Annotated, Any, Callable, get_origin

from semantikon.converter import FunctionWithMetadata, parse_metadata
from semantikon.datastructure import (
    MISSING,
    CoreMetadata,
    ExplicitDefault,
    Missing,
    RestrictionLike,
    ShapeType,
    TriplesLike,
    TypeMetadata,
)

__author__ = "Sam Waseda"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sam Waseda"
__email__ = "waseda@mpie.de"
__status__ = "development"
__date__ = "Aug 21, 2021"


def _is_annotated(type_):
    return hasattr(type_, "__metadata__") and hasattr(type_, "__origin__")


def _type_metadata(
    type_,
    /,
    uri: str | Missing = MISSING,
    triples: TriplesLike | Missing = MISSING,
    restrictions: RestrictionLike | Missing = MISSING,
    label: str | Missing = MISSING,
    units: str | Missing = MISSING,
    shape: ShapeType | Missing = MISSING,
    derived_from: str | Missing = MISSING,
    version: str | Missing = MISSING,
    **extra,
) -> Any:
    presently_requested_metadata = TypeMetadata(
        uri=uri,
        triples=triples,
        restrictions=restrictions,
        label=label,
        units=units,
        shape=shape,
        derived_from=derived_from,
        version=version,
    )

    kwargs = {"extra": extra} if len(extra) > 0 else {}
    if _is_annotated(type_):
        existing = parse_metadata(type_)
        if isinstance(existing.extra, dict):  # I.e., Not MISSING
            extra.update(existing.extra)
        kwargs.update(existing.to_dictionary())
        type_ = type_.__origin__
    kwargs.update(presently_requested_metadata.to_dictionary())
    if len(kwargs) == 0:
        raise TypeError("No metadata provided.")

    metadata: TypeMetadata = TypeMetadata.from_dict(kwargs)
    items = tuple([x for k, v in metadata for x in [k, v]])
    return Annotated[type_, items]


def _function_metadata(
    uri: str | Missing = MISSING,
    triples: TriplesLike | Missing = MISSING,
    restrictions: RestrictionLike | Missing = MISSING,
    version: str | Missing = MISSING,
):
    def decorator(func: Callable):
        if not callable(func):
            raise TypeError(f"Expected a callable, got {type(func)}")
        return FunctionWithMetadata(
            func,
            CoreMetadata(
                triples=triples,
                uri=uri,
                restrictions=restrictions,
                version=version
            ).to_dictionary(),
        )

    return decorator


def u(
    type_or_func=None,
    /,
    uri: str | Missing = MISSING,
    triples: TriplesLike | Missing = MISSING,
    restrictions: RestrictionLike | Missing = MISSING,
    label: str | Missing = MISSING,
    units: str | Missing = MISSING,
    shape: ShapeType | Missing = MISSING,
    derived_from: str | Missing = MISSING,
    version: str | Missing = MISSING,
    **kwargs,
) -> Callable[[Callable], FunctionWithMetadata] | Annotated[Any, object]:
    if isinstance(type_or_func, type) or get_origin(type_or_func) is not None:
        return _type_metadata(
            type_or_func,
            uri=uri,
            triples=triples,
            restrictions=restrictions,
            label=label,
            units=units,
            shape=shape,
            derived_from=derived_from,
            version=version,
            **kwargs,
        )
    elif type_or_func is None:
        if len(kwargs) > 0:
            raise NotImplementedError(
                "Function decoration does not currently support arbitrary keyword data."
            )
        return _function_metadata(
            uri=uri, triples=triples, restrictions=restrictions, version=version
        )
    else:
        raise TypeError(f"Unsupported type: {type(type_or_func)}")


def use_default(default: Any, msg: str | None = None) -> ExplicitDefault:
    """
    Marks a value as an explicit default, which can be used to indicate that
    a value should be replaced with a default value in the context of
    serialization or processing.

    Args:
        default (Any): The default value to be used.
        msg (str | None): An optional warning message. If not provided, the
            default message "'{arg}' not provided, using default: {value}" is
            used.

    Returns:
        ExplicitDefault: An instance of ExplicitDefault containing the default
        value and the message.

    Example:

    >>> @with_explicit_defaults
    >>> def f(x=use_default(3)):
    ...     return x

    >>> f()  # This will return 3, and a warning will be issued.

    >>> f(3)  # This will also return 3 but without any warning.
    """
    return ExplicitDefault(default, msg)
