import unittest
from dataclasses import asdict, astuple, is_dataclass
from typing import Annotated

from semantikon.converter import dataclass, parse_metadata
from semantikon.metadata import u


@dataclass
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


@dataclass
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
        @dataclass
        class BasicClass:
            x: int
            y: str

        # Create an instance of the class
        instance = BasicClass(x=10, y="hello")

        # Check if the instance attributes are set correctly
        self.assertEqual(instance.x, 10)
        self.assertEqual(instance.y, "hello")

        # Check if the instance is a dataclass
        self.assertTrue(is_dataclass(instance))

        # Check if the instance can be converted to a dictionary
        self.assertEqual(asdict(instance), {"x": 10, "y": "hello"})

        # Check if the instance can be converted to a tuple
        self.assertEqual(astuple(instance), (10, "hello"))

    def test_instance_equality(self):
        @dataclass
        class BasicClass:
            x: int
            y: str

        # Create two instances with the same values
        instance1 = BasicClass(x=10, y="hello")
        instance2 = BasicClass(x=10, y="hello")

        # Create an instance with different values
        instance3 = BasicClass(x=20, y="world")

        # Check equality
        self.assertEqual(instance1, instance2)
        self.assertNotEqual(instance1, instance3)

    def test_nested_class_instance(self):
        @dataclass
        class OuterClass:
            a: float

            @dataclass
            class InnerClass:
                b: Annotated[int, "integer"]

        # Create instances of the outer and inner classes
        outer_instance = OuterClass(a=3.14)
        inner_instance = OuterClass.InnerClass(b=42)

        # Check if the instances are dataclasses
        self.assertTrue(is_dataclass(outer_instance))
        self.assertTrue(is_dataclass(inner_instance))

        # Check if the attributes are set correctly
        self.assertEqual(outer_instance.a, 3.14)
        self.assertEqual(inner_instance.b, 42)

    def test_inheritance_instance(self):
        @dataclass
        class Parent:
            p: str

        @dataclass
        class Child(Parent):
            c: int

        # Create an instance of the child class
        child_instance = Child(p="parent", c=100)

        # Check if the instance is a dataclass
        self.assertTrue(is_dataclass(child_instance))

        # Check if the attributes are set correctly
        self.assertEqual(child_instance.p, "parent")
        self.assertEqual(child_instance.c, 100)

        # Check equality
        another_child_instance = Child(p="parent", c=100)
        self.assertEqual(child_instance, another_child_instance)

    def test_mutability(self):
        @dataclass
        class MutableClass:
            x: int
            y: str

        # Create an instance
        instance = MutableClass(x=10, y="hello")

        # Modify the attributes
        instance.x = 20
        instance.y = "world"

        # Check if the attributes are updated correctly
        self.assertEqual(instance.x, 20)
        self.assertEqual(instance.y, "world")

    def test_default_values(self):
        @dataclass
        class DefaultValuesClass:
            x: int = 42
            y: str = "default"

        # Create an instance without providing values
        instance = DefaultValuesClass()

        # Check if the default values are set correctly
        self.assertEqual(instance.x, 42)
        self.assertEqual(instance.y, "default")

        # Create an instance with custom values
        custom_instance = DefaultValuesClass(x=100, y="custom")

        # Check if the custom values are set correctly
        self.assertEqual(custom_instance.x, 100)
        self.assertEqual(custom_instance.y, "custom")


if __name__ == "__main__":
    unittest.main()
