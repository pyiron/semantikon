import unittest
from uniton.converter import append_types
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


class TestDataclass(unittest.TestCase):
    def setUp(self):
        append_types(Pizza)

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


if __name__ == "__main__":
    unittest.main()
