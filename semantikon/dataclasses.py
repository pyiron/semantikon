import abc
import dataclasses
import functools
from collections.abc import Iterable
from typing import Any, Iterator, Self, TypeAlias

import typeguard


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
        yield from (
            (f.name, val)
            for f in dataclasses.fields(self)
            if (val := getattr(self, f.name)) is not MISSING
        )

    @classmethod
    def from_dict(cls, kwargs: dict[str, Any]) -> Self:
        """Type-guarded instantiation from a dictionary"""

        for field in dataclasses.fields(cls):
            if field.name in kwargs:
                typeguard.check_type(kwargs[field.name], field.type)
        return cls(**kwargs)


TripleType: TypeAlias = tuple[str | None, str, str | None] | tuple[str, str]
TriplesLike: TypeAlias = tuple[TripleType, ...] | TripleType
RestrictionType: TypeAlias = tuple[str, str]
RestrictionLike: TypeAlias = (
    tuple[tuple[RestrictionType, ...], ...]
    | tuple[RestrictionType, ...]
    | RestrictionType
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
