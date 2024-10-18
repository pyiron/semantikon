import unittest
from uniton.typing import u
from uniton.converter import parse_input_args, parse_output_args


class TestUnits(unittest.TestCase):
    def test_use_list(self):
        def get_speed(
            distance: u(float, "meter", use_list=True),
            time: u(float, "second", use_list=True),
        ) -> u(float, "meter/second", use_list=True):
            return distance / time


if __name__ == "__main__":
    unittest.main()
