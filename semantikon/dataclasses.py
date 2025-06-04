import abc
import dataclasses
import functools
from collections.abc import Iterable, MutableMapping
from typing import Any, Callable, Generic, Iterator, TypeAlias, TypeVar

import typeguard


class Missing:
    def __repr__(self):
        return "<MISSING>"


MISSING = Missing()
missing = functools.partial(dataclasses.field, default=MISSING)


class _HasToDictionary(Iterable[tuple[str, Any]], abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[tuple[str, Any]]:
        pass

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
        yield from (
            (f.name, val)
            for f in dataclasses.fields(self)
            if (val := getattr(self, f.name)) is not MISSING
        )

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]):  # -> typing.Self only available 3.11+
        """Type-guarded instantiation from a dictionary"""

        for field in dataclasses.fields(cls):
            if field.name in kwargs:
                typeguard.check_type(kwargs[field.name], field.type)
        return cls(**kwargs)


TripleType: TypeAlias = tuple[str | None, str, str | None] | tuple[str, str]
TriplesLike: TypeAlias = tuple[TripleType, ...] | TripleType
RestrictionClause: TypeAlias = tuple[str, str]
RestrictionType: TypeAlias = tuple[RestrictionClause, ...]
RestrictionLike: TypeAlias = (
    tuple[RestrictionType, ...]  # Multiple restrictions
    | RestrictionType
    | RestrictionClause  # Short-hand for a single-clause restriction
)
ShapeType: TypeAlias = tuple[int, ...]


@dataclasses.dataclass(slots=True)
class CoreMetadata(_VariadicDataclass):
    uri: str | Missing = missing()
    triples: TriplesLike | Missing = missing()
    restrictions: RestrictionLike | Missing = missing()


@dataclasses.dataclass(slots=True)
class TypeMetadata(CoreMetadata):
    label: str | Missing = missing()
    units: str | Missing = missing()
    shape: ShapeType | Missing = missing()
    extra: dict[str, Any] | Missing = missing()


@dataclasses.dataclass(slots=True)
class _Port(TypeMetadata):
    dtype: type | Missing = missing()
    value: object | Missing = missing()


@dataclasses.dataclass(slots=True)
class Output(_Port):
    pass


@dataclasses.dataclass(slots=True)
class Input(_Port):
    default: Any | Missing = missing()


_ItemType = TypeVar("_ItemType")


class _HasToDictionarMapping(
    _HasToDictionary, MutableMapping[str, _ItemType], Generic[_ItemType]
):
    def __init__(self, **kwargs: _ItemType) -> None:
        self._data: dict[str, _ItemType] = kwargs

    def __getitem__(self, key: str) -> _ItemType:
        return self._data[key]

    def __setitem__(self, key: str, value: _ItemType) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[tuple[str, _ItemType]]:
        yield from self._data.items()

    def __len__(self) -> int:
        return len(self._data)


_PortType = TypeVar("_PortType", bound=_Port)


class _IO(_HasToDictionarMapping[_PortType], Generic[_PortType]): ...


class Inputs(_IO[Input]): ...


class Outputs(_IO[Output]): ...


@dataclasses.dataclass(slots=True)
class _Node(_VariadicDataclass):
    label: str
    inputs: Inputs
    outputs: Outputs
    metadata: CoreMetadata | Missing

    @property
    def type(self) -> str:
        return self.__class__.__name__

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        yield "type", self.type
        yield from super(_Node, self).__iter__()


@dataclasses.dataclass(slots=True)
class Function(_Node):
    function: Callable


class Nodes(_HasToDictionarMapping[_Node]): ...


EdgeType: TypeAlias = tuple[str, str]


class Edges(_HasToDictionarMapping[EdgeType]):
    """
    Key value pairs are stored as `{target: source}` such that each upstream source can
    be used in multiple places, but each downstream target can have only a single
    source.
    The :meth:`to_tuple` routine offers this reversed so that the returned tuples read
    intuitively as `(source, target)`.
    """

    def to_tuple(self) -> tuple[EdgeType, ...]:
        return tuple((e[1], e[0]) for e in self)


@dataclasses.dataclass(slots=True)
class Workflow(_Node):
    nodes: Nodes
    edges: Edges


@dataclasses.dataclass(slots=True)
class While(Workflow):
    test: _Node


@dataclasses.dataclass(slots=True)
class For(Workflow): ...  # TODO


@dataclasses.dataclass(slots=True)
class If(Workflow): ...  # TODO
