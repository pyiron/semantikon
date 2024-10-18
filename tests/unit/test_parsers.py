import unittest
from uniton.typing import u
from uniton.converter import parse_input_args, parse_output_args
import inspect


class TestUnits(unittest.TestCase):
    def test_basic(self):
        for use_list in [True, False]:
            def get_speed(
                distance: u(float, "meter", use_list=use_list),
                time: u(float, "second", use_list=use_list),
            ) -> u(float, "meter/second", label="speed", use_list=use_list):
                return distance / time
            input_args = parse_input_args(get_speed)
            for key in ["distance", "time"]:
                self.assertTrue(key in input_args)
            for key in ["units", "uri", "shape", "label", "dtype"]:
                self.assertTrue(key in input_args["distance"])
            self.assertEqual(input_args["distance"]["units"], "meter")
            self.assertEqual(input_args["time"]["units"], "second")
            output_args = parse_output_args(get_speed)
            for key in ["units", "uri", "shape", "label", "dtype"]:
                self.assertTrue(key in output_args)
            self.assertEqual(output_args["units"], "meter/second")
            self.assertEqual(output_args["label"], "speed")


if __name__ == "__main__":
    unittest.main()
