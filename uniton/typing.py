from typing import Annotated, Any

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
    otype: Any = None,
    shape: tuple[int] | None = None,
    use_list: bool = False,
    **kwargs,
):
    if use_list:
        if len(kwargs) > 0:
            raise ValueError("kwargs are not allowed when use_list=True")
        return Annotated[type_, units, otype, shape]
    else:
        result = {"units": units, "otype": otype, "shape": shape}
        result.update(kwargs)
        return Annotated[type_, str(result)]
