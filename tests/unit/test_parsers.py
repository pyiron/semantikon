import unittest
from semantikon.typing import u
from semantikon.converter import parse_input_args, parse_output_args, parse_metadata


class TestUnits(unittest.TestCase):
    def test_basic(self):
        for use_list in [True, False]:

            @u(uri="abc")
            def get_speed(
                distance: u(float, units="meter", use_list=use_list),
                time: u(float, units="second", use_list=use_list),
            ) -> u(float, units="meter/second", label="speed", use_list=use_list):
                return distance / time

            input_args = parse_input_args(get_speed)
            for key in ["distance", "time"]:
                self.assertTrue(key in input_args)
            for key in [
                "units",
                "uri",
                "triples",
                "shape",
                "label",
                "restrictions",
                "dtype"
            ]:
                self.assertTrue(key in input_args["distance"])
            self.assertEqual(input_args["distance"]["units"], "meter")
            self.assertEqual(input_args["time"]["units"], "second")
            output_args = parse_output_args(get_speed)
            for key in [
                "units",
                "uri",
                "triples",
                "shape",
                "label",
                "restrictions",
                "dtype"
            ]:
                self.assertTrue(key in output_args)
            self.assertEqual(output_args["units"], "meter/second")
            self.assertEqual(output_args["label"], "speed")
            self.assertEqual(get_speed._semantikon_metadata["uri"], "abc")
            self.assertRaises(TypeError, u)

    def test_multiple_output_args(self):
        for use_list in [True, False]:

            def get_speed(
                distance: u(float, units="meter", use_list=use_list),
                time: u(float, units="second", use_list=use_list),
            ) -> tuple[
                u(float, units="meter/second", label="speed", use_list=use_list),
                u(float, units="meter", label="distance", use_list=use_list),
            ]:
                return distance / time, distance

            output_args = parse_output_args(get_speed)
            self.assertIsInstance(output_args, tuple)
            for output_arg in output_args:
                for key in [
                    "units",
                    "uri",
                    "triples",
                    "shape",
                    "label",
                    "restrictions",
                    "dtype"
                ]:
                    self.assertTrue(key in output_arg)
            self.assertEqual(output_args[0]["units"], "meter/second")
            self.assertEqual(output_args[0]["label"], "speed")
            self.assertEqual(output_args[1]["units"], "meter")
            self.assertEqual(output_args[1]["label"], "distance")

    def test_additional_args(self):
        def get_speed(
            distance: u(float, units="meter", my_arg="some_info"),
            time: u(float, units="second"),
        ) -> u(float, units="meter/second", label="speed"):
            return distance / time

        input_args = parse_input_args(get_speed)
        self.assertEqual(input_args["distance"]["my_arg"], "some_info")

    def test_return_class(self):
        class Output:
            value: u(float, units="meter/second", label="speed")

        def get_speed(
            distance: u(float, units="meter"),
            time: u(float, units="second"),
        ) -> Output:
            return distance / time

        output_args = parse_output_args(get_speed)
        self.assertIsInstance(output_args, dict)
        self.assertEqual(output_args["dtype"], Output)

    def test_multiple_u(self):
        initial_type = u(float, units="meter", label="distance")
        result = parse_metadata(initial_type)
        self.assertEqual(result["units"], "meter")
        self.assertEqual(result["label"], "distance")
        final_type = u(initial_type, units="millimeter")
        result = parse_metadata(final_type)
        self.assertEqual(result["units"], "millimeter")
        self.assertEqual(result["label"], "distance")


if __name__ == "__main__":
    unittest.main()
