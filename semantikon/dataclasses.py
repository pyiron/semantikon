import dataclasses
import functools
from types import FunctionType
from typing import Any


class Missing:
    def __repr__(self):
        return "<MISSING>"


MISSING = Missing()
missing = functools.partial(dataclasses.field, default=MISSING)


@dataclasses.dataclass(slots=True)
class _VariadicDataclass:
    def asdict(self) -> dict[str, Any]:
        return {k: v for (k, v) in dataclasses.asdict(self).items() if v is not MISSING}


@dataclasses.dataclass(slots=True)
class _Port(_VariadicDataclass):
    dtype: type | Missing = missing()


@dataclasses.dataclass(slots=True)
class Output(_Port):
    pass


@dataclasses.dataclass(slots=True)
class Input(_Port):
    default: Any | Missing = missing()


@dataclasses.dataclass(slots=True)
class _Node(_VariadicDataclass):
    label: str
    inputs: dict[str, Input]
    outputs: dict[str, Output]

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
