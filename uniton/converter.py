# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pint import Quantity
import inspect
from functools import wraps
from typing import get_type_hints
from pint.registry_helpers import _apply_defaults, _parse_wrap_args, _to_units_container, _replace_units

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


def _get_ureg(args, kwargs):
    for arg in args + tuple(kwargs.values()):
        if isinstance(arg, Quantity):
            return arg._REGISTRY
    return None


def _get_args(sig):
    args = []
    for value in sig.parameters.values():
        if hasattr(value.annotation, "__metadata__"):
            args.append(value.annotation.__metadata__[0])
        else:
            args.append(None)
    return args

def _get_converter(sig):
    args = _get_args(sig)
    if any([arg is not None for arg in args]):
        return _parse_wrap_args(args)
    else:
        return None


def units(func):
    """
    Decorator to convert the output of a function to a Quantity object with the specified units.

    Args:
        func: function to be decorated

    Returns:
        decorated function
    """
    sig = inspect.signature(func)
    converter = _get_converter(sig)
    @wraps(func)
    def wrapper(*args, **kwargs):
        ureg = _get_ureg(args, kwargs)
        if converter is None or ureg is None:
            return func(*args, **kwargs)
        args, kwargs = _apply_defaults(sig, args, kwargs)
        args, kwargs, names = converter(ureg, sig, args, kwargs, strict=False)
        try:
            ret = _to_units_container(sig.return_annotation.__metadata__[0], ureg)
            out_units = (
                _replace_units(r, names) if is_ref else r
                for (r, is_ref) in ret
            )
            output_units = ureg.Quantity(1, _replace_units(ret[0], names) if ret[1] else ret[0])
        except AttributeError:
            output_units = None
        if output_units is None:
            return func(*args, **kwargs)
        else:
            return output_units * func(*args, **kwargs)
    return wrapper
