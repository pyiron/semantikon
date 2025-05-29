import abc
import dataclasses
import functools
from types import FunctionType
from typing import Any, ItemsView, TypeAlias


class _HasToDictionary(abc.ABC):
    @abc.abstractmethod
    def items(self) -> ItemsView[str, Any]: ...

    def to_dictionary(self) -> dict[str, Any]:
        d = {}
        for k, v in self.items():
            if isinstance(v, _HasToDictionary):
                d[k] = v.to_dictionary()
            else:
                d[k] = v
        return d


class Missing:
    def __repr__(self):
        return "<MISSING>"


MISSING = Missing()
missing = functools.partial(dataclasses.field, default=MISSING)


@dataclasses.dataclass(slots=True)
class _VariadicDataclass(_HasToDictionary):
    def asdict(self) -> dict[str, Any]:
        return {k: v for (k, v) in dataclasses.asdict(self).items() if v is not MISSING}

    def items(self) -> ItemsView[str, Any]:
        return self.asdict().items()


@dataclasses.dataclass(slots=True)
class _Port(_VariadicDataclass):
    dtype: type | Missing = missing()


@dataclasses.dataclass(slots=True)
class Output(_Port):
    pass


@dataclasses.dataclass(slots=True)
class Input(_Port):
    default: Any | Missing = missing()


class Inputs(dict[str, Input], _HasToDictionary): ...


class Outputs(dict[str, Output], _HasToDictionary): ...


@dataclasses.dataclass(slots=True)
class _Node(_VariadicDataclass):
    label: str
    inputs: Inputs
    outputs: Outputs

    @property
    def type(self) -> str:
        return self.__class__.__name__

    def asdict(self) -> dict[str, Any]:
        d = super().asdict()
        d["type"] = self.type
        return d

    def to_dictionary(self) -> dict[str, Any]:
        d = {}
        for k, v in self.asdict().items():
            if isinstance(v, _Node):
                d[k] = v.to_dictionary()
            elif isinstance(v, _VariadicDataclass):
                d[k] = v.asdict()
            else:
                d[k] = v
        return d


@dataclasses.dataclass(slots=True)
class Function(_Node):
    function: FunctionType


@dataclasses.dataclass(slots=True)
class Workflow(_Node):
    nodes: dict[str, _Node]
    edges: dict[str, tuple[str, str]]


@dataclasses.dataclass(slots=True)
class While(Workflow):
    test: _Node


@dataclasses.dataclass(slots=True)
class For(Workflow):
    raise NotImplementedError


@dataclasses.dataclass(slots=True)
class If(Workflow):
    raise NotImplementedError
