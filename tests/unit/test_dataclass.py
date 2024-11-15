import unittest
from uniton.converter import append_types, parse_metadata
from uniton.typing import u
from typing import Annotated
from dataclasses import dataclass


@dataclass
class Pizza:
    size: Annotated[float, "centimeter"]
    price: int = 30

    @dataclass
    class Topping:
        sauce: str

    def calculate_price(self, discount: float = 0) -> float:
        return self.price * (1 - discount)

    @property
    def original_price(self):
        return self.price


@dataclass
class Output:
    total_energy: u(float, units="eV", label="TotalEnergy", associate_to_sample=True)


class TestDataclass(unittest.TestCase):
    def setUp(self):
        append_types(Pizza)
        append_types(Output)

    def test_type(self):
        self.assertEqual(Pizza.price, int)
        self.assertEqual(Pizza.size, Annotated[float, "centimeter"])
        self.assertEqual(Pizza.Topping.sauce, str)

    def test_after_instantiation(self):
        pizza = Pizza(20)
        self.assertEqual(pizza.size, 20)
        self.assertEqual(pizza.price, 30)
        self.assertIsInstance(pizza, Pizza)
        sauce = Pizza.Topping("tomato")
        self.assertEqual(sauce.sauce, "tomato")
        self.assertIsInstance(sauce, Pizza.Topping)

    def test_parse_metadata(self):
        metadata = parse_metadata(Output.total_energy)
        self.assertEqual(
            metadata,
            {
                "units": "eV",
                "label": "TotalEnergy",
                "associate_to_sample": True,
                "shape": None,
                "uri": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
