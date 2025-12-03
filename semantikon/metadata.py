from typing import Annotated, Any, Callable, Optional
from uuid import uuid4

from rdflib import BNode, URIRef

from semantikon.converter import FunctionWithMetadata, parse_metadata
from semantikon.datastructure import (
    MISSING,
    FunctionMetadata,
    Missing,
    RestrictionLike,
    ShapeType,
    TriplesLike,
    TypeMetadata,
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


def _is_annotated(type_):
    return hasattr(type_, "__metadata__") and hasattr(type_, "__origin__")


def u(
    type_,
    /,
    uri: str | URIRef | Missing = MISSING,
    triples: TriplesLike | Missing = MISSING,
    restrictions: RestrictionLike | Missing = MISSING,
    label: str | URIRef | Missing = MISSING,
    units: str | URIRef | Missing = MISSING,
    shape: ShapeType | Missing = MISSING,
    derived_from: str | Missing = MISSING,
    **extra,
) -> Any:
    presently_requested_metadata = TypeMetadata(
        uri=uri,
        triples=triples,
        restrictions=restrictions,
        label=label,
        units=units,
        shape=shape,
        derived_from=derived_from,
    )

    kwargs = {"extra": extra} if len(extra) > 0 else {}
    if _is_annotated(type_):
        existing = parse_metadata(type_)
        if isinstance(existing.extra, dict):  # I.e., Not MISSING
            extra.update(existing.extra)
        kwargs.update(existing.to_dictionary())
        type_ = type_.__origin__
    kwargs.update(presently_requested_metadata.to_dictionary())
    if len(kwargs) == 0:
        raise TypeError("No metadata provided.")

    metadata: TypeMetadata = TypeMetadata.from_dict(kwargs)
    items = tuple([x for k, v in metadata for x in [k, v]])
    return Annotated[type_, items]


def meta(
    uri: str | URIRef | Missing = MISSING,
    triples: TriplesLike | Missing = MISSING,
    used: str | URIRef | Missing = MISSING,
):
    def decorator(func: Callable):
        if not callable(func):
            raise TypeError(f"Expected a callable, got {type(func)}")
        return FunctionWithMetadata(
            func,
            FunctionMetadata(triples=triples, uri=uri, used=used).to_dictionary(),
        )

    return decorator


class SemantikonURI(URIRef):
    """A class representing a URIRef with an associated blank node instance."""

    def __init__(
        self, value: str, base: Optional[str] = None, instance: Optional[BNode] = None
    ):
        """
        Initialize the SemantikonURI with a URIRef and an optional blank node instance.

        Args:
            value (str): The URI string.
            base (Optional[str]): The base URI for relative URIs.
            instance (Optional[BNode]): An optional blank node instance. If not
                provided, a new BNode is created.
        """
        self._uriref = URIRef(value=value, base=base)
        if instance is None:
            tag = value.split("/")[-1].split("#")[-1] + "_" + str(uuid4())
            instance = BNode(tag)
        self._instance = instance

    def get_class(self) -> URIRef:
        return self._uriref

    def get_instance(self) -> BNode:
        return self._instance
