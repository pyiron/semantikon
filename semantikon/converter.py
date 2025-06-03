# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import ast
import inspect
import re
import sys
import textwrap
from copy import deepcopy
from functools import update_wrapper, wraps
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from pint import Quantity, UnitRegistry
from pint.registry_helpers import (
    _apply_defaults,
    _parse_wrap_args,
    _replace_units,
    _to_units_container,
)

from semantikon.dataclasses import TypeMetadata

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


class NotAstNameError(TypeError): ...


F = TypeVar("F", bound=Callable[..., object])


class FunctionWithMetadata(Generic[F]):
    """
    A wrapper class for functions that allows attaching metadata to the function.

    Example:
    >>> from semantikon.typing import u
    >>>
    >>> @u(uri="http://example.com/my_function")
    >>> def my_function(x):
    >>>     return x * 2
    >>>
    >>> print(my_function._semantikon_metadata)

    Output: {'uri': 'http://example.com/my_function'}

    This information is automatically parsed when knowledge graph is generated.
    For more info, take a look at semantikon.ontology.get_knowledge_graph.
    """

    def __init__(self, func: F, metadata: dict[str, object]) -> None:
        self.func = func
        self._semantikon_metadata: dict[str, object] = metadata
        update_wrapper(self, func)  # Copies __name__, __doc__, etc.

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.func, item)

    def __deepcopy__(self, memo=None):
        new_func = deepcopy(self.func, memo)
        return FunctionWithMetadata(new_func, self._semantikon_metadata)


def _get_ureg(args: Any, kwargs: dict[str, Any]) -> UnitRegistry | None:
    for arg in args + tuple(kwargs.values()):
        if isinstance(arg, Quantity):
            return arg._REGISTRY
    return None


def parse_metadata(value: Any) -> TypeMetadata:
    """
    Parse the metadata of a Quantity object.

    Args:
        value: Quantity object

    Returns:
        dictionary of the metadata. Available keys are `units`, `label`,
        `triples`, `uri` and `shape`. See `semantikon.dataclasses.TypeMetadata` for more details.
    """
    metadata = value.__metadata__[0]
    return TypeMetadata(**{k: v for k, v in zip(metadata[::2], metadata[1::2])})


def meta_to_dict(value: Any, default=inspect.Parameter.empty) -> dict[str, Any]:
    semantikon_was_used = hasattr(value, "__metadata__")
    type_hint_was_present = value is not inspect.Parameter.empty
    default_is_defined = default is not inspect.Parameter.empty
    if semantikon_was_used:
        result = parse_metadata(value).to_dictionary()
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


def extract_undefined_name(error_message: str) -> str:
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


def _to_tag(item: Any, count=None, must_be_named: bool = False) -> str:
    if isinstance(item, ast.Name):
        return item.id
    elif must_be_named:
        raise NotAstNameError(
            "With `must_be_named=True`, item must be captured in an `ast.Name` variables."
        )
    elif count is None:
        return "output"
    else:
        return f"output_{count}"


def get_return_expressions(
    func: Callable, separate_tuple: bool = True, strict: bool = False
) -> str | tuple[str, ...] | None:
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    parsed = ast.parse(source)

    func_node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef))

    ret_list: list[str | tuple[str, ...]] = []

    for node in ast.walk(func_node):
        if isinstance(node, ast.Return):
            value = node.value
            if value is None:
                ret_list.append("None")
            elif isinstance(value, ast.Tuple):
                ret_list.append(
                    tuple(
                        [
                            _to_tag(elt, ii, must_be_named=strict)
                            for ii, elt in enumerate(value.elts)
                        ]
                    )
                )
            else:
                ret_list.append(_to_tag(value, must_be_named=strict))

    if len(ret_list) == 0 and not strict:
        return None
    elif len(set(ret_list)) == 1 and (
        separate_tuple or not isinstance(ret_list[0], tuple)
    ):
        return ret_list[0]
    elif (
        all(isinstance(exp, tuple) for exp in ret_list)
        and len(set(len(r) for r in ret_list)) == 1
        and separate_tuple
        and not strict
    ):
        return tuple([f"output_{i}" for i in range(len(ret_list[0]))])
    else:
        if strict:
            raise NotAstNameError(
                "With `strict=True`, all returns must be captured in independent "
                "variables."
            )
        return "output"


def get_return_labels(
    func: Callable, separate_tuple: bool = True, strict: bool = False
) -> tuple[str, ...]:
    return_vars = get_return_expressions(
        func, separate_tuple=separate_tuple, strict=strict
    )
    if return_vars is None:
        return ("None",)
    elif isinstance(return_vars, str):
        return (return_vars,)
    elif isinstance(return_vars, tuple) and all(
        isinstance(v, str) for v in return_vars
    ):
        return return_vars
    raise TypeError(
        f"{get_return_labels.__module__}.{get_return_labels.__qualname__} expected "
        f"None, a string, or a tuple of strings, but got {return_vars}"
    )


def get_annotated_type_hints(func: Callable) -> dict[str, Any]:
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


def parse_input_args(func: Callable) -> dict[str, dict]:
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


def parse_output_args(func: Callable, separate_tuple: bool = True) -> dict | tuple:
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
    if get_origin(ret) is tuple and separate_tuple:
        return tuple([meta_to_dict(ann) for ann in get_args(ret)])
    else:
        return meta_to_dict(ret)


def _get_converter(func: Callable) -> Callable | None:
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


def _get_ret_units(
    output: dict, ureg: UnitRegistry, names: dict[str, Any]
) -> Quantity | None:
    if output == {}:
        return None
    ret = _to_units_container(output.get("units", None), ureg)
    names = {key: 1.0 * value.units for key, value in names.items()}
    return ureg.Quantity(1, _replace_units(ret[0], names) if ret[1] else ret[0])


def _get_output_units(
    output: dict | tuple, ureg: UnitRegistry, names: dict[str, Any]
) -> Quantity | tuple[Quantity | None, ...] | None:
    if isinstance(output, tuple):
        return tuple([_get_ret_units(oo, ureg, names) for oo in output])
    else:
        return _get_ret_units(output, ureg, names)


def _is_dimensionless(output: Quantity | tuple[Quantity, ...] | None) -> bool:
    if output is None:
        return True
    if isinstance(output, tuple):
        return all([_is_dimensionless(oo) for oo in output])
    if output.to_base_units().magnitude == 1.0 and output.dimensionless:
        return True
    return False


def units(func: Callable) -> Callable:
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


def get_function_dict(function: Callable | FunctionWithMetadata) -> dict[str, Any]:
    result = {
        "label": function.__name__,
    }
    if hasattr(function, "_semantikon_metadata"):
        result.update(function._semantikon_metadata)
    return result


def semantikon_class(cls: type) -> type:
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
    setattr(cls, "_is_semantikon_class", True)
    return cls
