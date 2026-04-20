from typing import Annotated, Any, Callable, Optional
from uuid import uuid4

from rdflib import BNode, URIRef

from semantikon.converter import parse_metadata
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
    """
    A function that takes a type and metadata and returns an Annotated type
    with the metadata attached.

    Args:
        type_: Data type (e.g. int, float)
        uri: The URI associated with the argument.
        triples: RDF triples associated with the argument (cf. below)
        restrictions: Restrictions associated with the argument. Only to be
            used for input arguments, and must consist of doubles without blank
            nodes.
        label: A human-readable label for the argument.
        units: Units associated with the argument, if applicable. Either a
            string (such as "meter" or "meter**2/second") or a URIRef pointing
            to a unit in an ontology. If a string is provided, the unit will
            be parsed using the pint library, and the resulting unit will be
            translated to the corresponding URI in the QUDT ontology. If a
            URIRef is provided, it will be used
        shape: The shape of the data, if applicable - currently not used
        derived_from: Information about what this argument is derived from, if
            applicable. Only used for output arguments, and should be a string
            describing the derivation (e.g. "inputs.x").

    Returns:
        An Annotated type with the metadata attached.

    The `triples` argument can be either a double or a triple. The following
    annotations are all equivalent:

    >>> f(x: float) -> u(float, triples=(EX.outputOf, "inputs.x"))
    >>> f(x: float) -> u(float, triples=("self", EX.outputOf, "inputs.x"))
    >>> f(x: float) -> u(float, triples=(None, EX.outputOf, "inputs.x"))

    with an appropriate namespace `EX`. The `x` in `inputs.x` refers to the
    input variable.

    The `restrictions` argument must consist of doubles without the blank
    nodes and it can be used only for the input arguments, e.g.:

    >>> f(
    ...     x: u(
    ...         float, restrictions=(
    ...             (OWL.onProperty, EX.hasProperty),
    ...             (OWL.someValuesFrom, EX.SomeClass)
    ...         )
    ...     )
    ... ) -> float
    """
    units = extra.pop("unit", units)
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
    """
    A decorator for functions that allows attaching metadata to the function.

    Example:
    >>> from semantikon import meta
    >>>
    >>> @meta(uri="http://example.com/my_function")
    >>> def my_function(x):
    >>>     return x * 2
    >>>
    >>> print(my_function._semantikon_metadata)

    Output: {'uri': 'http://example.com/my_function'}

    This information is automatically parsed when knowledge graph is generated.
    For more info, take a look at semantikon.ontology.get_knowledge_graph.
    """

    def decorator(func: Callable):
        if not callable(func):
            raise TypeError(f"Expected a callable, got {type(func)}")
        metadata = FunctionMetadata(uri=uri, triples=triples, used=used).to_dictionary()

        func._semantikon_metadata = metadata  # type: ignore[attr-defined]
        return func

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
