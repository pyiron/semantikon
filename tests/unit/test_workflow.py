import unittest

from semantikon.workflow import number_to_letter


class TestSnippets(unittest.TestCase):
    def test_number_to_letter(self):
        self.assertEqual(number_to_letter(0), "A")
        self.assertEqual(number_to_letter(1), "B")


if __name__ == "__main__":
    unittest.main()
