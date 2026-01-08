import importlib.metadata

try:
    # Installed package will find its version
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # Repository clones will register an unknown version
    __version__ = "0.0.0+unknown"


from semantikon.api import (
    SemantikonURI,
    SparqlWriter,
    get_knowledge_graph,
    meta,
    query_io_completer,
    semantikon_dataclass,
    u,
    validate_values,
)

__all__ = [
    "SemantikonURI",
    "SparqlWriter",
    "semantikon_dataclass",
    "get_knowledge_graph",
    "meta",
    "query_io_completer",
    "u",
    "validate_values",
]
