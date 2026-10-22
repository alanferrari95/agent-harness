import unittest

from shop.pagination import page_count, page_slice


class TestPagination(unittest.TestCase):
    def test_first_page_starts_at_first_item(self):
        self.assertEqual(page_slice(list(range(10)), 1, 3), [0, 1, 2])

    def test_last_partial_page(self):
        self.assertEqual(page_slice(list(range(10)), 4, 3), [9])

    def test_page_count(self):
        self.assertEqual(page_count(10, 3), 4)
