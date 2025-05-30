from typing import Annotated, Any, Callable, get_origin

from semantikon.converter import FunctionWithMetadata, parse_metadata
from semantikon.dataclasses import (
    RestrictionLike,
    ShapeType,
    TriplesLike,
    CoreMetadata,
    TypeMetadata,
    Missing,
    MISSING,
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
    **kwargs,
) -> Any:
    presently_requested_metadata = TypeMetadata(
        uri=uri,
        triples=triples,
        restrictions=restrictions,
        label=label,
        units=units,
        shape=shape,
    )
    if _is_annotated(type_):
        kwargs.update(parse_metadata(type_))
        type_ = type_.__origin__
    kwargs.update(**presently_requested_metadata.to_dictionary())
    if len(kwargs) == 0:
        raise TypeError("No metadata provided.")
    metadata = TypeMetadata(**kwargs)
    items = tuple([x for k, v in metadata.to_dictionary().items() for x in [k, v]])
    return Annotated[type_, items]


def _function_metadata(
    uri: str | Missing = MISSING,
    triples: TriplesLike | Missing = MISSING,
    restrictions: RestrictionLike | Missing = MISSING,
):

    def decorator(func: Callable):
        if not callable(func):
            raise TypeError(f"Expected a callable, got {type(func)}")
        return FunctionWithMetadata(
            func,
            CoreMetadata(
                triples=triples, uri=uri, restrictions=restrictions
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
    **kwargs,
) -> Callable[[Callable], FunctionWithMetadata] | Annotated:
    if isinstance(type_or_func, type) or get_origin(type_or_func) is not None:
        return _type_metadata(
            type_or_func,
            uri=uri,
            triples=triples,
            restrictions=restrictions,
            label=label,
            units=units,
            shape=shape,
            **kwargs,
        )
    elif type_or_func is None:
        return _function_metadata(uri=uri, triples=triples, restrictions=restrictions)
    else:
        raise TypeError(f"Unsupported type: {type(type_or_func)}")
