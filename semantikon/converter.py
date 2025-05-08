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
from typing import get_origin, get_args, get_type_hints
import sys

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
        dictionary of the metadata. Available keys are `units`, `label`,
        `triples`, `uri` and `shape`. See `semantikon.typing.u` for more details.
    """
    metadata = value.__metadata__[0]
    return {k: v for k, v in zip(metadata[::2], metadata[1::2])}


def meta_to_dict(value, default=inspect.Parameter.empty):
    semantikon_was_used = hasattr(value, "__metadata__")
    type_hint_was_present = value is not inspect.Parameter.empty
    default_is_defined = default is not inspect.Parameter.empty
    if semantikon_was_used:
        result = {k: v for k, v in parse_metadata(value).items() if v is not None}
        result["dtype"] = value.__args__[0]
    else:
        result = {}
        if type_hint_was_present:
            result["dtype"] = value
    if default_is_defined:
        result["default"] = default
    return result


def get_annotated_type_hints(func):
    """
    Get the type hints of a function, including lazy annotations. The function
    practically does the same as `get_type_hints` for Python 3.11 and later,

    Args:
        func: function to be parsed

    Returns:
        dictionary of the type hints. The keys are the names of the arguments
        and the values are the type hints. The return type is stored under the
        key "return".
    """
    try:
        if sys.version_info >= (3, 11):
            # Use the official, public API
            return get_type_hints(func, include_extras=True)
        else:
            # Manually inspect __annotations__ and resolve them
            hints = {}
            sig = inspect.signature(func)
            for name, param in sig.parameters.items():
                annotation = param.annotation
                if isinstance(annotation, str):
                    # Lazy annotations: evaluate manually
                    annotation = eval(annotation, func.__globals__)
                hints[name] = annotation
            if sig.return_annotation is not inspect.Signature.empty:
                hints["return"] = sig.return_annotation
            return hints
    except NameError:
        hints = {}
        for key, value in func.__annotations__.items():
            try:
                hints[key] = eval(value, func.__globals__)
            except NameError:
                hints[key] = value
        if hasattr(func, "__return_annotation__"):
            hints["return"] = func.__return_annotation__
        return hints


def parse_input_args(func: callable):
    """
    Parse the input arguments of a function.

    Args:
        func: function to be parsed

    Returns:
        dictionary of the input arguments. Available keys are `units`, `label`,
        `triples`, `uri` and `shape`. See `semantikon.typing.u` for more details.
    """
    type_hints = get_annotated_type_hints(func)
    return {
        key: meta_to_dict(type_hints.get(key, value.annotation), value.default)
        for key, value in inspect.signature(func).parameters.items()
    }


def parse_output_args(func: callable):
    """
    Parse the output arguments of a function.

    Args:
        func: function to be parsed

    Returns:
        dictionary of the output arguments if there is only one output. Otherwise,
        a list of dictionaries is returned. Available keys are `units`,
        `label`, `triples`, `uri` and `shape`. See `semantikon.typing.u` for
        more details.
    """
    ret = get_annotated_type_hints(func).get("return", inspect.Parameter.empty)
    multiple_output = get_origin(ret) is tuple
    if multiple_output:
        return tuple([meta_to_dict(ann) for ann in get_args(ret)])
    else:
        return meta_to_dict(ret)


def _get_converter(func):
    args = []
    for value in parse_input_args(func).values():
        if value is not None:
            args.append(value.get("units", None))
        else:
            args.append(None)
    if any([arg is not None for arg in args]):
        return _parse_wrap_args(args)
    else:
        return None


def _get_ret_units(output, ureg, names):
    if output == {}:
        return None
    ret = _to_units_container(output.get("units", None), ureg)
    names = {key: 1.0 * value.units for key, value in names.items()}
    return ureg.Quantity(1, _replace_units(ret[0], names) if ret[1] else ret[0])


def _get_output_units(output, ureg, names):
    multiple_output_args = isinstance(output, tuple)
    if multiple_output_args:
        return tuple([_get_ret_units(oo, ureg, names) for oo in output])
    else:
        return _get_ret_units(output, ureg, names)


def _is_dimensionless(output):
    if output is None:
        return True
    if isinstance(output, tuple):
        return all([_is_dimensionless(oo) for oo in output])
    if output.to_base_units().magnitude == 1.0 and output.dimensionless:
        return True
    return False


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

        # Extend kwargs to account for **kwargs
        ext_kwargs = {
            key: kwargs.get(key, 0) for key in list(sig.parameters.keys())[len(args) :]
        }

        args, new_kwargs, names = converter(ureg, sig, args, ext_kwargs, strict=False)
        for key in list(new_kwargs.keys()):
            if key not in kwargs:
                new_kwargs.pop(key)

        try:
            output_units = _get_output_units(parse_output_args(func), ureg, names)
        except AttributeError:
            output_units = None

        if _is_dimensionless(output_units):
            return func(*args, **new_kwargs)
        elif isinstance(output_units, tuple):
            return tuple(
                [oo * ff for oo, ff in zip(output_units, func(*args, **new_kwargs))]
            )
        else:
            return output_units * func(*args, **new_kwargs)

    return wrapper


def get_function_dict(function):
    result = {
        "label": function.__name__,
    }
    function_has_metadata = hasattr(function, "_semantikon_metadata")
    if function_has_metadata:
        result.update(function._semantikon_metadata)
    return result


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
    cls._is_semantikon_class = True
    return cls
