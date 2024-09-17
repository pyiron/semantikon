import numpy as np
import unittest
from uniton.typing import u
from uniton.converter import units
from pint import UnitRegistry


@units
def get_speed_ints(
    distance: u(int, "meter"), time: u(int, "second")
) -> u(int, "meter/second"):
    return distance / time


@units
def get_speed_floats(
    distance: u(float, "meter"), time: u(float, "second")
) -> u(float, "meter/second"):
    return distance / time


@units
def get_speed_relative(
    distance: u(float, "=A"), time: u(float, "=B")
) -> u(float, "=A/B"):
    return distance / time


class TestTools(unittest.TestCase):
    def test_relative(self):
        self.assertEqual(get_speed_relative(1, 1), 1)
        ureg = UnitRegistry()
        self.assertEqual(
            get_speed_relative(1 * ureg.angstrom, 1 * ureg.meter),
            1 * ureg.angstrom / ureg.meter,
        )

    def test_ints(self):
        self.assertEqual(get_speed_ints(1, 1), 1)
        ureg = UnitRegistry()
        self.assertEqual(
            get_speed_ints(1 * ureg.meter, 1 * ureg.second),
            1 * ureg.meter / ureg.second,
        )
        self.assertEqual(
            get_speed_ints(1 * ureg.millimeter, 1 * ureg.second),
            0.001 * ureg.meter / ureg.second,
        )


if __name__ == "__main__":
    unittest.main()
