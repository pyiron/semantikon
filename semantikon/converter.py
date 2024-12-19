# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pint import Quantity
import inspect
from functools import wraps
from pint.registry_helpers import (
    _apply_defaults,
    _parse_wrap_args,
    _to_units_container,
    _replace_units,
)
from ast import literal_eval
from typing import get_origin, get_args

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


def parse_metadata(value):
    """
    Parse the metadata of a Quantity object.

    Args:
        value: Quantity object

    Returns:
        dictionary of the metadata. Available keys are `units`, `otype`,
        `shape`, and `dtype`. See `semantikon.typing.u` for more details.
    """
    # When there is only one metadata `use_list=False` must have been used
    if len(value.__metadata__) == 1 and isinstance(value.__metadata__[0], str):
        return literal_eval(value.__metadata__[0])
    else:
        d = {}
        for ii in range(len(value.__metadata__[0]) // 2):
            d[value.__metadata__[0][2 * ii]] = value.__metadata__[0][2 * ii + 1]
        return d


def _meta_to_dict(value):
    if hasattr(value, "__metadata__"):
        result = parse_metadata(value)
        result["dtype"] = value.__args__[0]
        return result
    elif value is not inspect.Parameter.empty:
        return {
            "units": None,
            "label": None,
            "uri": None,
            "shape": None,
            "dtype": value,
        }
    else:
        return None


def parse_input_args(func: callable):
    """
    Parse the input arguments of a function.

    Args:
        func: function to be parsed

    Returns:
        dictionary of the input arguments. Available keys are `units`, `otype`,
        and `shape`. See `semantikon.typing.u` for more details.
    """
    return {
        key: _meta_to_dict(value.annotation)
        for key, value in inspect.signature(func).parameters.items()
    }


def parse_output_args(func: callable):
    """
    Parse the output arguments of a function.

    Args:
        func: function to be parsed

    Returns:
        dictionary of the output arguments if there is only one output. Otherwise,
        a list of dictionaries is returned. Available keys are `units`, `otype`,
        and `shape`. See `semantikon.typing.u` for more details.
    """
    sig = inspect.signature(func)
    if get_origin(sig.return_annotation) is tuple:
        return tuple([_meta_to_dict(ann) for ann in get_args(sig.return_annotation)])
    else:
        return _meta_to_dict(sig.return_annotation)


def _get_converter(func):
    args = []
    for value in parse_input_args(func).values():
        if value is not None:
            args.append(value["units"])
        else:
            args.append(None)
    if any([arg is not None for arg in args]):
        return _parse_wrap_args(args)
    else:
        return None


def _get_ret_units(output, ureg, names):
    if output is None:
        return None
    ret = _to_units_container(output["units"], ureg)
    return ureg.Quantity(1, _replace_units(ret[0], names) if ret[1] else ret[0])


def _get_output_units(output, ureg, names):
    if isinstance(output, tuple):
        return tuple([_get_ret_units(oo, ureg, names) for oo in output])
    else:
        return _get_ret_units(output, ureg, names)


def units(func):
    """
    Decorator to convert the output of a function to a Quantity object with
    the specified units.

    Args:
        func: function to be decorated

    Returns:
        decorated function
    """
    sig = inspect.signature(func)
    converter = _get_converter(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        ureg = _get_ureg(args, kwargs)
        if converter is None or ureg is None:
            return func(*args, **kwargs)
        args, kwargs = _apply_defaults(sig, args, kwargs)
        args, kwargs, names = converter(ureg, sig, args, kwargs, strict=False)
        try:
            output_units = _get_output_units(parse_output_args(func), ureg, names)
        except AttributeError:
            output_units = None
        if output_units is None:
            return func(*args, **kwargs)
        elif isinstance(output_units, tuple):
            return tuple(
                [oo * ff for oo, ff in zip(output_units, func(*args, **kwargs))]
            )
        else:
            return output_units * func(*args, **kwargs)

    return wrapper


def semantikon_class(cls: type):
    """
    A class decorator to append type hints to class attributes.

    Args:
        cls: class to be decorated

    Returns:
        The modified class with type hints appended to its attributes.

    Comments:

    >>> from typing import Annotated
    >>> from semantikon.converter import semantikon_class

    >>> @semantikon_class
    >>> class Pizza:
    >>>     price: Annotated[float, "money"]
    >>>     size: Annotated[float, "dimension"]

    >>>     class Topping:
    >>>         sauce: Annotated[str, "matter"]

    >>> append_types(Pizza)
    >>> print(Pizza)
    >>> print(Pizza.Topping)
    >>> print(Pizza.size)
    >>> print(Pizza.price)
    >>> print(Pizza.Topping.sauce)
    """
    for key, value in cls.__dict__.items():
        if isinstance(value, type):
            semantikon_class(getattr(cls, key))  # Recursively apply to nested classes
    try:
        for key, value in cls.__annotations__.items():
            setattr(cls, key, value)  # Append type hints to attributes
    except AttributeError:
        pass
    return cls
