import abc
import dataclasses
import functools
from collections.abc import Iterable, MutableMapping
from typing import Any, Generic, Iterator, TypeAlias, TypeVar


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


TripleType: TypeAlias = tuple[str, str, str]
TriplesLike: TypeAlias = tuple[TripleType, ...] | TripleType
RestrictionType: TypeAlias = tuple[str, str]
RestrictionLike: TypeAlias = tuple[RestrictionType, ...] | RestrictionType
ShapeType: TypeAlias = tuple[int, ...]


@dataclasses.dataclass(slots=True)
class CoreMetadata(_VariadicDataclass):
    # Drawn from signature of semantikon.typing._function_metadata
    uri: str | Missing = missing()
    triples: TriplesLike | Missing = missing()
    restrictions: RestrictionLike | Missing = missing()


@dataclasses.dataclass(slots=True)
class TypeMetadata(CoreMetadata):
    # Drawn from signature of semantikon.typing._type_metadata
    label: str | Missing = missing()
    units: str | Missing = missing()
    shape: ShapeType | Missing = missing()

    # Stuff that gets passed to _type_metadata during the tests
    associate_to_sample: bool | Missing = missing()
    cancel: tuple[Any, Any] | Missing = missing()
    my_arg: str | Missing = missing()
    use_list: bool | Missing = missing()
    # How should we handle this?


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
        # yield "type", self.__class__.__name__  # Disabled for backwards compatibility
        yield from super(_Node, self).__iter__()


@dataclasses.dataclass(slots=True)
class Function(
    _Node
):
    # function: FunctionType  # Disabled for backwards compatibility
    uri: str | Missing = missing()  # Ad-hoc addition to satisfy the `add` test


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
