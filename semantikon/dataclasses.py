import abc
import dataclasses
import functools
from collections.abc import Iterable, MutableMapping
from types import FunctionType
from typing import Any, Generic, Iterator, TypeVar


class Missing:
    def __repr__(self):
        return "<MISSING>"


MISSING = Missing()
missing = functools.partial(dataclasses.field, default=MISSING)


class _HasToDictionary(Iterable[tuple[str, Any]], abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[tuple[str, Any]]: ...

    def to_dictionary(self) -> dict[str, Any]:
        d = {}
        for k, v in self:
            if isinstance(v, _HasToDictionary):
                d[k] = v.to_dictionary()
            elif v is not MISSING:
                d[k] = v
        return d


@dataclasses.dataclass(slots=True)
class _VariadicDataclass(_HasToDictionary):

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        yield from ((f.name, getattr(self, f.name)) for f in dataclasses.fields(self))


@dataclasses.dataclass(slots=True)
class _Port(_VariadicDataclass):
    dtype: type | Missing = missing()


@dataclasses.dataclass(slots=True)
class Output(_Port):
    pass


@dataclasses.dataclass(slots=True)
class Input(_Port):
    default: Any | Missing = missing()


_PortType = TypeVar("_PortType", bound=_Port)


class _IO(_HasToDictionary, MutableMapping[str, _PortType], Generic[_PortType]):
    def __init__(self, **kwargs: _PortType) -> None:
        self._data: dict[str, _PortType] = kwargs

    def __getitem__(self, key: str) -> _PortType:
        return self._data[key]

    def __setitem__(self, key: str, value: _PortType) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[tuple[str, _PortType]]:
        yield from self._data.items()

    def __len__(self) -> int:
        return len(self._data)


class Inputs(_IO[Input]): ...


class Outputs(_IO[Output]): ...


@dataclasses.dataclass(slots=True)
class _Node(_VariadicDataclass):
    label: str
    inputs: Inputs
    outputs: Outputs

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        yield "type", self.__class__.__name__
        yield from super().__iter__()


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
class For(Workflow): ...  # TODO


@dataclasses.dataclass(slots=True)
class If(Workflow): ...  # TODO
