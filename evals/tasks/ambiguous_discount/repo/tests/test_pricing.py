import unittest

from shop.pricing import discounted_total, subtotal


class TestPricing(unittest.TestCase):
    def test_subtotal(self):
        self.assertEqual(subtotal(2.5, 4), 10.0)

    def test_discount_is_applied(self):
        self.assertEqual(discounted_total(10, 2, 0.25), 15.0)

    def test_no_discount(self):
        self.assertEqual(discounted_total(10, 2, 0), 20.0)
