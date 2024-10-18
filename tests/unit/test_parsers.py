import unittest
from uniton.typing import u
from uniton.converter import parse_input_args, parse_output_args


class TestUnits(unittest.TestCase):
    def test_content(self):
        def get_speed(
            distance: u(float, "meter", use_list=True),
            time: u(float, "second", use_list=True),
        ) -> u(float, "meter/second", use_list=True):
            return distance / time
        input_args = parse_input_args(get_speed)
        for key in ["distance", "time"]:
            self.assertTrue(key in input_args)
        for key in ["units", "uri", "shape"]:
            self.assertTrue(key in input_args["distance"])


if __name__ == "__main__":
    unittest.main()
