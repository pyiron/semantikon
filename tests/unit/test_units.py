import unittest
from semantikon.typing import u
from semantikon.converter import units
from pint import UnitRegistry


@units
def get_speed_multiple_outputs(
    distance: u(float, units="meter"),
    time: u(float, units="second"),
    duration: u(float, units="second"),
) -> tuple[u(float, units="meter/second"), u(float, units="meter/second")]:
    return distance / time, distance / duration


@units
def get_speed_no_output_type(
    distance: u(float, units="meter"), time: u(float, units="second")
):
    return distance / time


@units
def get_speed_multiple_args(
    distance: u(float, units="meter"),
    time: u(float, units="second"),
    duration: u(float | None, units="second") = None,
) -> u(float, units="meter/second"):
    if duration is None:
        return distance / time
    else:
        return distance / duration


@units
def get_speed_optional_args(
    distance: u(float, units="meter"), time: u(float, units="second") = 1
) -> u(float, units="meter/second"):
    return distance / time


@units
def get_speed_ints(
    distance: u(int, units="meter"), time: u(int, units="second")
) -> u(int, units="meter/second"):
    return distance / time


@units
def get_speed_floats(
    distance: u(float, units="meter"), time: u(float, units="second")
) -> u(float, units="meter/second"):
    return distance / time


@units
def get_speed_relative(
    distance: u(float, units="=A"), time: u(float, units="=B")
) -> u(float, units="=A/B"):
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
            distance: u(float, units="meter", use_list=False),
            time: u(float, units="second", use_list=False),
        ) -> u(float, units="meter/second", use_list=False):
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


if __name__ == "__main__":
    unittest.main()
