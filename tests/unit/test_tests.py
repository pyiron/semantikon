import unittest

import semantikon


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = semantikon.__version__
        print(version)
        self.assertTrue(version.startswith("0"))
