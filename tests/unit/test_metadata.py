"""Tests for :mod:`semantikon.metadata`."""

import copy
import pickle
import unittest

from rdflib import URIRef

from semantikon import metadata


class TestSemantikonURI(unittest.TestCase):
    """``SemantikonURI`` must survive copying and pickling.

    ``rdflib.URIRef.__reduce__`` hard-codes ``URIRef`` as the reconstruction
    class, so without an override a (deep)copy or unpickle of a
    ``SemantikonURI`` silently degrades to a plain ``URIRef`` and loses its
    blank-node instance. ``flowrep_dict`` now puts these into dicts that
    ``semantikon.workflow.to_semantikon_workflow_dict`` deep-copies, so the
    coercion would corrupt the ontology triples.
    """

    def setUp(self):
        self.uri = metadata.SemantikonURI("http://example.org/Color")

    def test_deepcopy_preserves_type(self):
        copied = copy.deepcopy(self.uri)
        self.assertIsInstance(copied, metadata.SemantikonURI)

    def test_copy_preserves_type(self):
        copied = copy.copy(self.uri)
        self.assertIsInstance(copied, metadata.SemantikonURI)

    def test_pickle_preserves_type(self):
        copied = pickle.loads(pickle.dumps(self.uri))
        self.assertIsInstance(copied, metadata.SemantikonURI)

    def test_deepcopy_preserves_uri_value(self):
        copied = copy.deepcopy(self.uri)
        self.assertEqual(str(copied), str(self.uri))
        self.assertEqual(copied.get_class(), self.uri.get_class())

    def test_deepcopy_preserves_instance(self):
        copied = copy.deepcopy(self.uri)
        self.assertEqual(copied.get_instance(), self.uri.get_instance())

    def test_deepcopy_inside_container_preserves_type(self):
        """The real failure mode: nested in the metadata dict structure."""
        data = {"triples": (URIRef("http://example.org/hasProperty"), self.uri)}
        copied = copy.deepcopy(data)
        self.assertIsInstance(copied["triples"][1], metadata.SemantikonURI)


if __name__ == "__main__":
    unittest.main()
