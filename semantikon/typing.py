from typing import Annotated, get_origin
from semantikon.converter import parse_metadata

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
    use_list: bool = True,
    **kwargs,
):
    parent_result = {}
    if _is_annotated(type_):
        parent_result = parse_metadata(type_)
        type_ = type_.__origin__
    result = {
        "units": units,
        "label": label,
        "triples": triples,
        "uri": uri,
        "shape": shape,
        "restrictions": restrictions,
    }
    for key, value in parent_result.items():
        if result[key] is None:
            result[key] = value
    result.update(kwargs)
    if use_list:
        items = [x for k, v in result.items() for x in [k, v]]
        return Annotated[type_, items]
    else:
        return Annotated[type_, str(result)]


def _function_metadata(
    triples: tuple[tuple[str, str, str], ...] | tuple[str, str, str] | None = None,
    uri: str | None = None,
    restrictions: tuple[tuple[str, str]] | None = None,
    **kwargs,
):
    data = {
        "triples": triples,
        "uri": uri,
        "restrictions": restrictions
    }
    data.update(kwargs)
    def decorator(func: callable):
        func._semantikon_metadata = data
        return func
    return decorator


def u(type_or_func=None, /, **kwargs):
    if isinstance(type_or_func, type) or get_origin(type_or_func) is not None:
        return _type_metadata(type_or_func, **kwargs)
    elif type_or_func is None:
        return _function_metadata(**kwargs)
    else:
        raise TypeError(f"Unsupported type: {type(type_or_func)}")
