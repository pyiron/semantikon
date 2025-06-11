import dataclasses
import unittest

import typeguard

import semantikon.dataclasses as sdc


@dataclasses.dataclass
class ConcreteDC(sdc._VariadicDataclass):
    complex_field: set[str]
    optional_field: int | sdc.Missing = sdc.missing()


class ConcreteHtDM(sdc._HasToDictionarMapping[int]): ...


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
            msg="Dataclass field should hold full data object, even missing objects",
        )
        self.assertNotIn(
            "optional_field",
            self.dc.to_dictionary(),
            msg="Sending the dataclass to a dictionary should purge missing entries --"
            "this is why we call it a variadic dataclass, because some of its "
            "fields are optional!",
        )

    def test_iter(self):
        self.assertIsNot(self.dc.complex_field, sdc.MISSING, msg="Sanity check")
        self.assertIs(self.dc.optional_field, sdc.MISSING, msg="Sanity check")
        self.assertEqual(
            2, len(dataclasses.fields(self.dc)), msg="Make sure we tested them all"
        )
        self.assertListEqual(
            [("complex_field", self.dc.complex_field)],
            [v for v in self.dc],
            msg="Iterating should exclude missing values",
        )

    def test_from_dict(self):
        ok = ConcreteDC.from_dict({"complex_field": self.dc.complex_field})
        self.assertSetEqual(ok.complex_field, self.dc.complex_field)

        with self.assertRaises(
            typeguard.TypeCheckError,
            msg="If we don't care about type compliance, we can already use __init__ "
            "to initialize from a dictionary; this method is explicitly to enforce "
            "type compliance!",
        ):
            ConcreteDC.from_dict({"complex_field": "This is not type-compliant"})


class TestHasToDictionaryMapping(unittest.TestCase):
    def test_mapping(self):
        t = (1, 2, 3)
        a, b, c = t
        mapping = ConcreteHtDM(a=a, b=b)
        self.assertEqual(mapping["a"], a)
        self.assertEqual(mapping["b"], b)

        mapping["c"] = c
        self.assertEqual(mapping.c, c)

        self.assertEqual(len(mapping), len(t))

        del mapping["b"]
        self.assertEqual(len(mapping), len(t) - 1)
        self.assertIsNone(mapping.get("b", None))
        self.assertEqual(mapping.a, a)
        self.assertEqual(mapping.c, c)


if __name__ == "__main__":
    unittest.main()
