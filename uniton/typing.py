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


def u(type_, /, units: str | None = None, otype: Any = None):
    return Annotated[type_, units, otype]
