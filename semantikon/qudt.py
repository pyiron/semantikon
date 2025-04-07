import re
import os
from rdflib import Graph, RDFS
from pint import UnitRegistry


def contains_special_char(s):
    # This pattern matches any character that is NOT a letter, digit, or underscore
    return bool(re.search(r'[^a-zA-Z0-9_./ ]', s))


class UnitsDict:
    def __init__(self, graph=None, location=None):
        if graph is None:
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
