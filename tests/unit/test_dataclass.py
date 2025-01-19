import unittest
from semantikon.typing import u
from typing import Annotated
from dataclasses import dataclass
from semantikon.converter import semantikon_class, parse_metadata


@semantikon_class
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


@semantikon_class
@dataclass
class Output:
    total_energy: u(float, units="eV", label="TotalEnergy", associate_to_sample=True)


class TestDataclass(unittest.TestCase):

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
                "triples": None,
                "shape": None,
                "uri": None,
                "restrictions": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
