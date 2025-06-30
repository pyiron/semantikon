import os
import re
from collections import defaultdict
from functools import cached_property

import requests
from pint import UnitRegistry
from rdflib import RDFS, Graph, URIRef, term


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

    @cached_property
    def _base_units(self) -> dict[str, term.Node]:
        data = defaultdict(list)
        for key in self._units_dict.keys():
            try:
                data[str(self._ureg[key.lower()].to_base_units().units)].append(key)
            except Exception:
                pass
        return data

    def __getitem__(self, key: str) -> term.Node | None:
        if key.startswith("http"):
            return URIRef(key)
        key = key.lower()
        if key in self._units_dict:
            return self._units_dict[key]
        key = str(self._ureg[str(key)].units)
        if key in self._units_dict:
            return self._units_dict[key]
        new_key = str(self._ureg[key].to_base_units().units)
        if new_key in self._base_units:
            raise KeyError(
                f"'{key}' is not available in QUDT; use a full URI. "
                "Alternatively, you can change it to one of the following units:"
                f"{', '.join(self._base_units[new_key])}"
            )
        raise KeyError(f"'{key}' is not available in QUDT; use a full URI.")


def get_graph(location: str | None = None) -> Graph:
    if location is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        location = os.path.join(script_dir, "data", "qudt.ttl")
    graph = Graph()
    graph.parse(location=location, format="ttl")
    return graph


def get_units_dict(graph: Graph) -> dict[str, term.Node]:
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
        tag = str(tag).lower()
        units_dict[tag] = uri
        for _ in range(2):
            try:
                # this is safe and works for both Quantity and Unit
                tag = str(ureg[str(tag).lower()].units)
                if tag not in units_dict or len(str(uri)) < len(str(units_dict[tag])):
                    units_dict[tag] = uri
            except Exception:
                pass
            # This line replaces e.g. "electron volt" by "electron_volt"
            tag = re.sub(r'(?<=[a-zA-Z]) (?=[a-zA-Z])', '_', tag)
    return units_dict
