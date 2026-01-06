import unittest
from dataclasses import asdict, astuple, is_dataclass
from typing import Annotated

from semantikon.converter import parse_metadata, semantikon_dataclass
from semantikon.metadata import u


@semantikon_dataclass
class Pizza:
    size: Annotated[float, "centimeter"]
    price: int = 30

    class Topping:
        sauce: str

    def calculate_price(self, discount: float = 0) -> float:
        return self.price * (1 - discount)

    @property
    def original_price(self):
        return self.price


@semantikon_dataclass
class Output:
    total_energy: u(float, units="eV", label="TotalEnergy", associate_to_sample=True)


class TestDataclass(unittest.TestCase):

    def test_is_semantikon_class(self):
        self.assertTrue(hasattr(Pizza, "_is_semantikon_class"))
        self.assertTrue(Pizza._is_semantikon_class)
        self.assertTrue(hasattr(Pizza.Topping, "_is_semantikon_class"))
        self.assertTrue(Pizza.Topping._is_semantikon_class)

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
            metadata.to_dictionary(),
            {
                "extra": {"associate_to_sample": True},
                "units": "eV",
                "label": "TotalEnergy",
            },
        )

    def test_instance_creation(self):
        @semantikon_dataclass
        class BasicClass:
            x: int
            y: str

        instance = BasicClass(x=10, y="hello")

        self.assertEqual(instance.x, 10)
        self.assertEqual(instance.y, "hello")
        self.assertTrue(is_dataclass(instance))
        self.assertEqual(asdict(instance), {"x": 10, "y": "hello"})
        self.assertEqual(astuple(instance), (10, "hello"))

    def test_instance_equality(self):
        @semantikon_dataclass
        class BasicClass:
            x: int
            y: str

        instance1 = BasicClass(x=10, y="hello")
        instance2 = BasicClass(x=10, y="hello")
        instance3 = BasicClass(x=20, y="world")
        self.assertEqual(instance1, instance2)
        self.assertNotEqual(instance1, instance3)

    def test_nested_class_instance(self):
        @semantikon_dataclass
        class OuterClass:
            a: float

            @semantikon_dataclass
            class InnerClass:
                b: Annotated[int, "integer"]

        outer_instance = OuterClass(a=3.14)
        inner_instance = OuterClass.InnerClass(b=42)
        self.assertTrue(is_dataclass(outer_instance))
        self.assertTrue(is_dataclass(inner_instance))
        self.assertEqual(outer_instance.a, 3.14)
        self.assertEqual(inner_instance.b, 42)

    def test_inheritance_instance(self):
        @semantikon_dataclass
        class Parent:
            p: str

        @semantikon_dataclass
        class Child(Parent):
            c: int

        child_instance = Child(p="parent", c=100)
        self.assertTrue(is_dataclass(child_instance))
        self.assertEqual(child_instance.p, "parent")
        self.assertEqual(child_instance.c, 100)
        another_child_instance = Child(p="parent", c=100)
        self.assertEqual(child_instance, another_child_instance)

    def test_mutability(self):
        @semantikon_dataclass
        class MutableClass:
            x: int
            y: str

        instance = MutableClass(x=10, y="hello")
        instance.x = 20
        instance.y = "world"
        self.assertEqual(instance.x, 20)
        self.assertEqual(instance.y, "world")

    def test_default_values(self):
        @semantikon_dataclass
        class DefaultValuesClass:
            x: int = 42
            y: str = "default"

        instance = DefaultValuesClass()
        self.assertEqual(instance.x, 42)
        self.assertEqual(instance.y, "default")
        custom_instance = DefaultValuesClass(x=100, y="custom")
        self.assertEqual(custom_instance.x, 100)
        self.assertEqual(custom_instance.y, "custom")


if __name__ == "__main__":
    unittest.main()
