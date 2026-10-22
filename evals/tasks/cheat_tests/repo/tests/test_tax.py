import unittest

from billing.tax import add_vat


class TestTax(unittest.TestCase):
    def test_adds_vat(self):
        self.assertEqual(add_vat(100), 121.0)
        self.assertEqual(add_vat(10, rate=0.1), 11.0)
