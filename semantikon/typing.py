from typing import Annotated, Any
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
    if hasattr(type_, "__metadata__"):
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


def _function_metadata(**kwargs):
    def decorator(func):
        func.metadata = kwargs
        return func
    return decorator


def u(*args, **kwargs):
    if isinstance(args[0], type) or hasattr(args[0], "__metadata__"):
        return _type_metadata(*args, **kwargs)
    elif callable(args[0]):
        if len(args) > 1:
            raise TypeError(f"Invalid type: {args}")
        return _function_metadata(*args, **kwargs)
    else:
        raise TypeError(f"Invalid type: {args}, {kwargs}")
