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


def u(
    type_,
    /,
    units: str | None = None,
    label: str | None = None,
    triple: tuple[tuple[str, str, str], ...] | tuple[str, str, str] | None = None,
    uri: str | None = None,
    shape: tuple[int] | None = None,
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
        "triple": triple,
        "uri": uri,
        "shape": shape,
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
