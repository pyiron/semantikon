# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import inspect
import re
import sys
from functools import wraps
from typing import Annotated, Callable, get_args, get_origin, get_type_hints

from pint import Quantity
from pint.registry_helpers import (
    _apply_defaults,
    _parse_wrap_args,
    _replace_units,
    _to_units_container,
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
        if hasattr(value.__args__[0], "__forward_arg__"):
            result["dtype"] = value.__args__[0].__forward_arg__
        else:
            result["dtype"] = value.__args__[0]
    else:
        result = {}
        if type_hint_was_present:
            result["dtype"] = value
    if default_is_defined:
        result["default"] = default
    return result


def extract_undefined_name(error_message):
    match = re.search(r"name '(.+?)' is not defined", error_message)
    if match:
        return match.group(1)
    raise ValueError(
        "No undefined name found in the error message: {}".format(error_message)
    )


def _resolve_annotation(annotation, func_globals=None):
    if func_globals is None:
        func_globals = globals()
    if not isinstance(annotation, str):
        return annotation
    # Lazy annotations: evaluate manually
    try:
        return eval(annotation, func_globals)
    except NameError as e:
        # Handle undefined names in lazy annotations
        undefined_name = extract_undefined_name(str(e))
        if undefined_name == annotation:
            return annotation
        new_annotations = eval(annotation, func_globals | {undefined_name: object})
        args = get_args(new_annotations)
        assert len(args) == 2, "Invalid annotation format"
        return Annotated[undefined_name, args[1]]


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
                hints[name] = _resolve_annotation(param.annotation, func.__globals__)
            if sig.return_annotation is not inspect.Signature.empty:
                hints["return"] = _resolve_annotation(
                    sig.return_annotation, func.__globals__
                )
            return hints
    except NameError:
        hints = {}
        for key, value in func.__annotations__.items():
            hints[key] = _resolve_annotation(value, func.__globals__)
        if hasattr(func, "__return_annotation__"):
            hints["return"] = _resolve_annotation(
                func.__return_annotation__, func.__globals__
            )
        return hints


def parse_input_args(func: Callable):
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


def parse_output_args(func: Callable):
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
