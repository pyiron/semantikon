import importlib.metadata

try:
    # Installed package will find its version
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # Repository clones will register an unknown version
    __version__ = "0.0.0+unknown"


from semantikon.api import (
    FunctionMetadata,
    SemantikonURI,
    TypeMetadata,
    annotation_to_type_hint,
    annotation_to_type_metadata,
    get_knowledge_graph,
    kg2data,
    kg2recipe,
    meta,
    parse_metadata,
    query_io_completer,
    request_values,
    semantikon_dataclass,
    u,
    validate_values,
    visualize_recipe,
)

__all__ = [
    "FunctionMetadata",
    "SemantikonURI",
    "TypeMetadata",
    "annotation_to_type_hint",
    "annotation_to_type_metadata",
    "get_knowledge_graph",
    "kg2data",
    "kg2recipe",
    "meta",
    "parse_metadata",
    "query_io_completer",
    "request_values",
    "semantikon_dataclass",
    "u",
    "validate_values",
    "visualize_recipe",
]
