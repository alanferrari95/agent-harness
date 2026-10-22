import unittest

from inventory.reorder import needs_reorder


class TestReorder(unittest.TestCase):
    def test_below_threshold(self):
        self.assertTrue(needs_reorder(3, 10))

    def test_at_threshold(self):
        self.assertTrue(needs_reorder(10, 10))

    def test_above_threshold(self):
        self.assertFalse(needs_reorder(11, 10))
