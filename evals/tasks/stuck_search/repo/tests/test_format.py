import unittest

from money.format import format_price


class TestFormat(unittest.TestCase):
    def test_two_decimals(self):
        self.assertEqual(format_price(3.5), "$3.50")
        self.assertEqual(format_price(2), "$2.00")
