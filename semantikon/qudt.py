import re
import os
from rdflib import Graph, RDFS
from pint import UnitRegistry
import requests


def download_data(version=None, store_data=False):
    if version is None:
        version = "3.1.0"
    data = requests.get(f"https://qudt.org/{version}/vocab/unit").text
    graph = Graph()
    graph.parse(data=data, format="ttl")
    graph_with_only_label = Graph()
    for s, p, o in graph:
        if p == RDFS.label:
            graph_with_only_label.add((s, p, o))
    if store_data:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        location = os.path.join(script_dir, 'data', 'qudt.ttl')
        graph_with_only_label.serialize(destination=location, format="ttl")
    return graph_with_only_label


class UnitsDict:
    def __init__(
        self,
        graph=None,
        location=None,
        force_download=False,
        version=None,
        store_data=False,
    ):
        if force_download:
            graph = download_data(version=version, store_data=store_data)
        elif graph is None:
            graph = get_graph(location)
        self._units_dict = get_units_dict(graph)
        self._ureg = UnitRegistry()

    def __getitem__(self, key):
        key = key.lower()
        if key in self._units_dict:
            return self._units_dict[key]
        key = str(self._ureg[str(key)].units)
        if key in self._units_dict:
            return self._units_dict[key]
        return None


def get_graph(location=None):
    if location is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        location = os.path.join(script_dir, 'data', 'qudt.ttl')
    graph = Graph()
    graph.parse(location=location, format="ttl")
    return graph


def get_units_dict(graph):
    ureg = UnitRegistry()
    units_dict = {}
    for uri, tag in graph.subject_objects(RDFS.label):
        if tag.language is not None and not tag.language.startswith("en"):
            continue
        try:
            key = str(ureg[str(tag).lower()].units)
        except:
            key = str(tag).lower()
        units_dict[key] = uri
    return units_dict
