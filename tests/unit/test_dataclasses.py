import dataclasses
import unittest

import semantikon.dataclasses as sdc


class TestDataclasses(unittest.TestCase):
    def test_item_identity(self):
        @dataclasses.dataclass
        class ConcreteDC(sdc._VariadicDataclass):
            complex_field: set[str]

        complex_object = {"Here is some non-trivial, mutable object"}
        dc = ConcreteDC(complex_field=complex_object)
        self.assertIs(
            dc.complex_field,
            complex_object,
            msg="Sanity check that we get instance we pass in",
        )
        self.assertIs(
            dc.complex_field,
            dc.to_dictionary()["complex_field"],
            msg="The dictionary representation should return the same underlying data",
        )
