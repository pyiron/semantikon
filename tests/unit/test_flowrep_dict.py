"""Tests for flowrep annotation conversion helpers."""

import unittest

from semantikon import datastructure, flowrep_dict
from semantikon.metadata import u


class TestAnnotationConverters(unittest.TestCase):
    def test_type_hint_none(self):
        self.assertIsNone(flowrep_dict.annotation_to_type_hint(None))

    def test_type_hint_plain(self):
        self.assertIs(flowrep_dict.annotation_to_type_hint(float), float)

    def test_type_hint_strips_annotated_to_base(self):
        self.assertIs(flowrep_dict.annotation_to_type_hint(u(int, units="meter")), int)

    def test_type_metadata_none(self):
        self.assertIsNone(flowrep_dict.annotation_to_type_metadata(None))

    def test_type_metadata_plain_is_none(self):
        self.assertIsNone(flowrep_dict.annotation_to_type_metadata(float))

    def test_type_metadata_from_annotated(self):
        meta = flowrep_dict.annotation_to_type_metadata(u(int, units="meter"))
        self.assertIsInstance(meta, datastructure.TypeMetadata)
        self.assertEqual(meta.units, "meter")
