import unittest

from semantikon.qudt import UnitsDict


class TestQUDT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ud = UnitsDict(force_download=True, store_data=True)

    def test_uri(self):
        meter = self.ud["meter"]
        self.assertEqual(meter, self.ud["m"])
        self.assertEqual(str(meter), "http://qudt.org/vocab/unit/M")
        self.assertEqual(str(self.ud["eV"]), "http://qudt.org/vocab/unit/EV")
        self.assertEqual(
            str(self.ud["Cubic Meter per Square Meter"]),
            "http://qudt.org/vocab/unit/M3-PER-M2",
        )
        self.assertEqual(
            str(self.ud["http://qudt.org/vocab/unit/M"]), "http://qudt.org/vocab/unit/M"
        )

    def test_graph_consistency(self):
        ud = UnitsDict()
        self.assertEqual(ud["second"], self.ud["second"])


if __name__ == "__main__":
    unittest.main()
