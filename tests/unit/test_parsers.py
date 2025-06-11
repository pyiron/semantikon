import unittest
from typing import TYPE_CHECKING
from unittest import mock

from semantikon.converter import (
    NotAstNameError,
    get_function_dict,
    get_return_expressions,
    get_return_labels,
    parse_input_args,
    parse_metadata,
    parse_output_args,
)
from semantikon.metadata import u

if TYPE_CHECKING:

    class Atoms:
        pass


class TestParser(unittest.TestCase):
    def test_basic(self):
        @u(uri="abc")
        def get_speed(
            distance: u(float, units="meter"),
            time: u(float, units="second") = 1.0,
        ) -> u(float, units="meter/second", label="speed"):
            return distance / time

        input_args = parse_input_args(get_speed)
        for key in ["distance", "time"]:
            self.assertIn(key, input_args)
        for key in ["units", "dtype"]:
            self.assertIn(key, input_args["distance"])
        self.assertEqual(input_args["distance"]["units"], "meter")
        self.assertEqual(input_args["time"]["default"], 1.0)
        self.assertEqual(input_args["time"]["units"], "second")
        output_args = parse_output_args(get_speed)
        for key in [
            "units",
            "dtype",
        ]:
            self.assertIn(key, output_args)
        self.assertEqual(output_args["units"], "meter/second")
        self.assertEqual(output_args["label"], "speed")
        self.assertEqual(get_speed._semantikon_metadata["uri"], "abc")
        self.assertRaises(TypeError, u, "abc")
        f_dict = get_function_dict(get_speed)
        self.assertEqual(f_dict["uri"], "abc")
        self.assertEqual(f_dict["label"], "get_speed")

    def test_extra_function_metadata(self):
        def f(x):
            return x

        with self.assertRaises(
            NotImplementedError,
            msg="Arbitrary metadata is not currently supported for function decoration",
        ):
            u(uri="abc", unexpected_data=123)(f)

    def test_canonical_types(self):
        def f(x: float) -> float:
            return x

        input_args = parse_input_args(f)
        self.assertEqual(input_args["x"]["dtype"], float)

    def test_multiple_output_args(self):
        def get_speed(
            distance: u(float, units="meter"),
            time: u(float, units="second"),
        ) -> tuple[
            u(float, units="meter/second", label="speed"),
            u(float, units="meter", label="distance"),
        ]:
            return distance / time, distance

        output_args = parse_output_args(get_speed)
        self.assertIsInstance(output_args, tuple)
        for output_arg in output_args:
            self.assertIn("dtype", output_arg)
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
        self.assertEqual(input_args["distance"]["extra"]["my_arg"], "some_info")

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
        result = parse_metadata(initial_type).to_dictionary()
        self.assertEqual(result["units"], "meter")
        self.assertEqual(result["label"], "distance")
        final_type = u(initial_type, units="millimeter")
        result = parse_metadata(final_type).to_dictionary()
        self.assertEqual(result["units"], "millimeter")
        self.assertEqual(result["label"], "distance")

    def test_invalid_u(self):
        with self.assertRaises(TypeError) as context:
            u(float)
        self.assertEqual(str(context.exception), "No metadata provided.")

    def test_optional_args(self):
        def get_speed_multiple_args(
            distance: u(float, units="meter"),
            time: u(float, units="second"),
            duration: u(float | None, units="second") = None,
        ) -> u(float, units="meter/second"):
            if duration is None:
                return distance / time
            else:
                return distance / duration

        input_args = parse_input_args(get_speed_multiple_args)
        for value, key in zip(input_args.values(), ["meter", "second", "second"]):
            self.assertEqual(value["units"], key)

    def test_future(self):
        def test_another_future(x: "Atoms", y: "u(float, units='second')") -> "Atoms":
            return x

        input_args = parse_input_args(test_another_future)
        self.assertEqual(input_args["x"]["dtype"], "Atoms")
        self.assertIn("units", input_args["y"])
        self.assertEqual(input_args["y"]["units"], "second")
        output_args = parse_output_args(test_another_future)
        self.assertEqual(output_args["dtype"], "Atoms")

    def test_semantikon_and_future(self):
        # This imitates a future import
        def test_more_future(x: "u(Atoms, uri='metadata')") -> "Atoms":
            return x

        input_args = parse_input_args(test_more_future)
        self.assertEqual(input_args["x"]["uri"], "metadata")
        self.assertEqual(input_args["x"]["dtype"], "Atoms")

    def test_get_return_expressions(self):
        def f(x):
            return x

        self.assertEqual(get_return_expressions(f), "x")
        self.assertEqual(get_return_expressions(f, separate_tuple=False), "x")
        self.assertEqual(get_return_expressions(f, strict=True), "x")
        self.assertEqual(
            get_return_expressions(f, separate_tuple=False, strict=True), "x"
        )

        def f(x, y):
            return x, y

        self.assertEqual(get_return_expressions(f), ("x", "y"))
        self.assertEqual(get_return_expressions(f, separate_tuple=False), "output")
        self.assertEqual(get_return_expressions(f, strict=True), ("x", "y"))
        with self.assertRaises(
            NotAstNameError,
            msg="These are always incompatible boolean kwargs with a tuple return",
        ):
            get_return_expressions(f, separate_tuple=False, strict=True)

        def f(x, y):
            return x, -y

        self.assertEqual(get_return_expressions(f), ("x", "output_1"))
        self.assertEqual(get_return_expressions(f, separate_tuple=False), "output")
        with self.assertRaises(NotAstNameError):
            get_return_expressions(f, strict=True)

        def f(x, y):
            if x < 0:
                return x, y
            else:
                return x, y

        self.assertEqual(get_return_expressions(f), ("x", "y"))
        self.assertEqual(get_return_expressions(f, separate_tuple=False), "output")
        self.assertEqual(get_return_expressions(f, strict=True), ("x", "y"))

        def f(x, y):
            if x < 0:
                return x, y
            else:
                return y, x

        self.assertEqual(get_return_expressions(f), ("output_0", "output_1"))
        self.assertEqual(get_return_expressions(f, separate_tuple=False), "output")
        with self.assertRaises(NotAstNameError):
            get_return_expressions(f, strict=True)

        def f(x, y):
            if x < 0:
                return x
            else:
                return y, x

        self.assertEqual(get_return_expressions(f), "output")
        self.assertEqual(get_return_expressions(f, separate_tuple=False), "output")
        with self.assertRaises(NotAstNameError):
            get_return_expressions(f, strict=True)

        def f(x):
            print("hello")

        self.assertIsNone(get_return_expressions(f))
        self.assertIsNone(get_return_expressions(f, separate_tuple=False))
        with self.assertRaises(NotAstNameError):
            get_return_expressions(f, strict=True)

        def f(x):
            return

        self.assertEqual(get_return_expressions(f), "None")
        self.assertEqual(get_return_expressions(f, separate_tuple=False), "None")
        self.assertEqual(get_return_expressions(f, strict=True), "None")

    def test_get_return_labels(self):

        def f(x):
            return x

        with mock.patch("semantikon.converter.get_return_expressions", return_value=123
        ):
            with self.assertRaises(
                TypeError, msg="expected None, a string, or a tuple of strings"
            ):
                get_return_labels(f)


if __name__ == "__main__":
    unittest.main()
