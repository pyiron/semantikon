import dataclasses
import unittest

import semantikon.dataclasses as sdc


@dataclasses.dataclass
class ConcreteDC(sdc._VariadicDataclass):
    complex_field: set[str]
    optional_field: int | sdc.Missing = sdc.missing()


class TestDataclasses(unittest.TestCase):
    def setUp(self):
        self.complex_object = {"Here is some non-trivial, mutable object"}
        self.dc = ConcreteDC(complex_field=self.complex_object)

    def test_item_identity(self):
        self.assertIs(
            self.dc.complex_field,
            self.complex_object,
            msg="Sanity check that we get instance we pass in",
        )
        self.assertIs(
            self.dc.complex_field,
            self.dc.to_dictionary()["complex_field"],
            msg="The dictionary representation should return the same underlying data",
        )

    def test_missing(self):
        self.assertIs(
            self.dc.optional_field,
            sdc.MISSING,
            msg="Dataclass field should hold full data object, even missing objects"
        )
        self.assertNotIn(
            "optional_field",
            self.dc.to_dictionary(),
            msg="Sending the dataclass to a dictionary should purge missing entries --"
                "this is why we call it a variadic dataclass, because some of its "
                "fields are optional!"
        )