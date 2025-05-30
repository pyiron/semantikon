from typing import Annotated, Any, Callable, get_origin

from semantikon.converter import FunctionWithMetadata, parse_metadata

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
    units: str | None = None,
    label: str | None = None,
    triples: tuple[tuple[str, str, str], ...] | tuple[str, str, str] | None = None,
    uri: str | None = None,
    shape: tuple[int] | None = None,
    restrictions: tuple[tuple[str, str]] | None = None,
    **kwargs,
) -> Any:
    if _is_annotated(type_):
        kwargs.update(parse_metadata(type_))
        type_ = type_.__origin__
    kwargs.update(
        _kwargs_to_dict(
            units=units,
            label=label,
            triples=triples,
            uri=uri,
            shape=shape,
            restrictions=restrictions,
        )
    )
    if len(kwargs) == 0:
        raise TypeError("No metadata provided.")
    items = tuple([x for k, v in kwargs.items() for x in [k, v]])
    return Annotated[type_, items]


def _function_metadata(
    triples: tuple[tuple[str, str, str], ...] | tuple[str, str, str] | None = None,
    uri: str | None = None,
    restrictions: tuple[tuple[str, str]] | None = None,
    **kwargs,
):
    data: dict[str, object] = {
        k: v
        for k, v in {
            "triples": triples,
            "uri": uri,
            "restrictions": restrictions,
        }.items()
        if v is not None
    }
    data.update(kwargs)
    for key, value in kwargs.items():
        if value is None:
            data.pop(key)

    def decorator(func: Callable):
        if not callable(func):
            raise TypeError(f"Expected a callable, got {type(func)}")
        return FunctionWithMetadata(func, data)

    return decorator


def _kwargs_to_dict(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


def u(
    type_or_func=None,
    /,
    units: str | None = None,
    label: str | None = None,
    triples: tuple[tuple[str, str, str], ...] | tuple[str, str, str] | None = None,
    uri: str | None = None,
    shape: tuple[int] | None = None,
    restrictions: tuple[tuple[str, str]] | None = None,
    **kwargs,
) -> Any:
    kwargs.update(
        _kwargs_to_dict(
            units=units,
            label=label,
            triples=triples,
            uri=uri,
            shape=shape,
            restrictions=restrictions,
        )
    )
    if isinstance(type_or_func, type) or get_origin(type_or_func) is not None:
        return _type_metadata(type_or_func, **kwargs)
    elif type_or_func is None:
        return _function_metadata(**kwargs)
    else:
        raise TypeError(f"Unsupported type: {type(type_or_func)}")
