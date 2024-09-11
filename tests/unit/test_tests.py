import unittest
import uniton


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = uniton.__version__
        print(version)
        self.assertTrue(version.startswith('0'))
