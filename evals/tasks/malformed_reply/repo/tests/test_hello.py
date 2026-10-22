import unittest

from greetings.hello import greet


class TestGreet(unittest.TestCase):
    def test_format(self):
        self.assertEqual(greet("Ada"), "Hello, Ada!")
