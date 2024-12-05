import unittest
from semantikon.typing import u
from semantikon.converter import units
from pint import UnitRegistry


@units
def get_speed_multiple_outputs(
    distance: u(float, "meter"), time: u(float, "second"), duration: u(float, "second")
) -> (u(float, "meter/second"), u(float, "meter/second")):
    return distance / time, distance / duration


@units
def get_speed_no_output_type(distance: u(float, "meter"), time: u(float, "second")):
    return distance / time


@units
def get_speed_multiple_args(
    distance: u(float, "meter"),
    time: u(float, "second"),
    duration: u(float | None, "second") = None,
) -> u(float, "meter/second"):
    if duration is None:
        return distance / time
    else:
        return distance / duration


@units
def get_speed_optional_args(
    distance: u(float, "meter"), time: u(float, "second") = 1
) -> u(float, "meter/second"):
    return distance / time


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


class TestUnits(unittest.TestCase):
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
            get_speed_ints(1 * ureg.meter, 1 * ureg.millisecond),
            1000 * ureg.meter / ureg.second,
        )

    def test_floats(self):
        self.assertEqual(get_speed_floats(1.0, 1.0), 1.0)
        ureg = UnitRegistry()
        self.assertEqual(
            get_speed_floats(1.0 * ureg.meter, 1.0 * ureg.second),
            1.0 * ureg.meter / ureg.second,
        )
        self.assertEqual(
            get_speed_floats(1.0 * ureg.millimeter, 1.0 * ureg.second),
            0.001 * ureg.meter / ureg.second,
        )

    def test_multiple_args(self):
        self.assertEqual(get_speed_multiple_args(1, 1), 1)
        ureg = UnitRegistry()
        self.assertEqual(
            get_speed_multiple_args(1 * ureg.meter, 1 * ureg.second),
            1 * ureg.meter / ureg.second,
        )
        self.assertEqual(
            get_speed_multiple_args(1 * ureg.meter, 1 * ureg.second, 1 * ureg.second),
            1 * ureg.meter / ureg.second,
        )
        self.assertEqual(
            get_speed_multiple_args(
                1 * ureg.meter, 1 * ureg.second, 1 * ureg.millisecond
            ),
            1000 * ureg.meter / ureg.second,
        )

    def test_optional_args(self):
        self.assertEqual(get_speed_optional_args(1), 1)
        ureg = UnitRegistry()
        self.assertEqual(
            get_speed_optional_args(1 * ureg.meter),
            1 * ureg.meter / ureg.second,
        )
        self.assertEqual(
            get_speed_optional_args(1 * ureg.meter, 1 * ureg.second),
            1 * ureg.meter / ureg.second,
        )
        self.assertEqual(
            get_speed_optional_args(1 * ureg.meter, 1 * ureg.millisecond),
            1000 * ureg.meter / ureg.second,
        )

    def test_no_output_type(self):
        self.assertEqual(get_speed_no_output_type(1, 1), 1)
        ureg = UnitRegistry()
        self.assertEqual(get_speed_no_output_type(1 * ureg.meter, 1 * ureg.second), 1)
        self.assertEqual(
            get_speed_no_output_type(1 * ureg.millimeter, 1 * ureg.second), 0.001
        )

    def test_multiple_outputs(self):
        self.assertEqual(get_speed_multiple_outputs(1, 1, 1), (1, 1))
        ureg = UnitRegistry()
        self.assertEqual(
            get_speed_multiple_outputs(
                1 * ureg.meter, 1 * ureg.second, 1 * ureg.second
            ),
            (1 * ureg.meter / ureg.second, 1 * ureg.meter / ureg.second),
        )
        self.assertEqual(
            get_speed_multiple_outputs(
                1 * ureg.meter, 1 * ureg.second, 1 * ureg.millisecond
            ),
            (1 * ureg.meter / ureg.second, 1000 * ureg.meter / ureg.second),
        )

    def test_use_list(self):
        @units
        def get_speed_use_list(
            distance: u(float, "meter", use_list=True),
            time: u(float, "second", use_list=True),
        ) -> u(float, "meter/second", use_list=True):
            return distance / time
        self.assertEqual(get_speed_use_list(1.0, 1.0), 1.0)
        ureg = UnitRegistry()
        self.assertEqual(
            get_speed_use_list(1.0 * ureg.meter, 1.0 * ureg.second),
            1.0 * ureg.meter / ureg.second,
        )
        self.assertEqual(
            get_speed_use_list(1.0 * ureg.millimeter, 1.0 * ureg.second),
            0.001 * ureg.meter / ureg.second,
        )
        with self.assertRaises(
            ValueError, msg="user keyword is not supported when use_list is True"
        ):
            _ = u(float, my_arg="ahoi", use_list=True)


if __name__ == "__main__":
    unittest.main()
