import os

import requests
from pint import UnitRegistry
from rdflib import RDFS, Graph, URIRef


def _is_english_label(pred, obj):
    return (
        pred == RDFS.label
        and obj.language is not None
        and obj.language.startswith("en")
    )


def _is_symbol(pred):
    return pred == URIRef("http://qudt.org/schema/qudt/symbol")


def download_data(version: str | None = None, store_data: bool = False) -> Graph:
    """
    Download the QUDT data from the QUDT website and parse it into a graph.

    Parameters
    ----------
    version : str, optional
        The version of QUDT to download. If None, the latest version will be
        downloaded.
    store_data : bool, optional
        If True, the downloaded data will be stored in a local file.

    Returns
    -------
    graph : rdflib.Graph
        The graph containing the QUDT data.
    """
    if version is None:
        version = "3.1.0"
    data = requests.get(f"https://qudt.org/{version}/vocab/unit", timeout=300).text
    graph = Graph()
    graph.parse(data=data, format="ttl")
    graph_with_only_label = Graph()
    for s, p, o in graph:
        if _is_english_label(p, o) or _is_symbol(p):
            graph_with_only_label.add((s, p, o))
    if store_data:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        location = os.path.join(script_dir, "data", "qudt.ttl")
        graph_with_only_label.serialize(destination=location, format="ttl")
    return graph_with_only_label


class UnitsDict:
    """
    A class to represent a dictionary of units.
    The dictionary is created from the QUDT data and can be used to
    convert between different units.
    """

    def __init__(
        self,
        graph=None,
        location=None,
        force_download=False,
        version=None,
        store_data=False,
    ):
        """
        Parameters
        ----------
        graph : rdflib.Graph, optional
            The graph containing the QUDT data. If None, the data will be
            downloaded from the QUDT website.
        location : str, optional
            The location of the QUDT data file. If None, the data file
            stored in the semantikon repository will be used.
        force_download : bool, optional
            If True, the data will be downloaded from the QUDT website
            even if the graph is provided.
        version : str, optional
            The version of QUDT to download. If None, the latest version
            will be downloaded.
        store_data : bool, optional
            If True, the downloaded data will be stored in a local file.
        """
        if force_download:
            graph = download_data(version=version, store_data=store_data)
        elif graph is None:
            graph = get_graph(location)
        self._units_dict = get_units_dict(graph)
        self._ureg = UnitRegistry()

    def __getitem__(self, key):
        if key.startswith("http"):
            return URIRef(key)
        key = key.lower()
        if key in self._units_dict:
            return self._units_dict[key]
        key = str(self._ureg[str(key)])
        if key in self._units_dict:
            return self._units_dict[key]


def get_graph(location: str | None = None) -> Graph:
    if location is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        location = os.path.join(script_dir, "data", "qudt.ttl")
    graph = Graph()
    graph.parse(location=location, format="ttl")
    return graph


def get_units_dict(graph: Graph) -> dict[str, URIRef]:
    """
    Create a dictionary of units from the QUDT data.

    Parameters
    ----------
    graph : rdflib.Graph
        The graph containing the QUDT data.

    Returns
    -------
    units_dict : dict
        A dictionary mapping unit names to their URIs.
    """
    ureg = UnitRegistry()
    units_dict = {}
    for uri, tag in graph.subject_objects(None):
        key = str(tag).lower()
        units_dict[key] = uri
        try:
            key = str(
                ureg[str(tag).lower()]
            )  # this is safe and works for both Quantity and Unit
            if key not in units_dict or len(str(uri)) < len(str(units_dict[key])):
                units_dict[key] = uri
        except Exception:
            pass
    return units_dict
