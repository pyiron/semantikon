import importlib.metadata

try:
    # Installed package will find its version
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # Repository clones will register an unknown version
    __version__ = "0.0.0+unknown"


from semantikon.api import SemantikonURI, get_knowledge_graph, meta, u, validate_values

__all__ = [
    "SemantikonURI",
    "get_knowledge_graph",
    "meta",
    "u",
    "validate_values",
]
