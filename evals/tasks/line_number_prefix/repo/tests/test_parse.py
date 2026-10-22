import unittest
from datetime import date

from dates.parse import parse_iso_date


class TestParse(unittest.TestCase):
    def test_iso_date(self):
        self.assertEqual(parse_iso_date("2026-10-22"), date(2026, 10, 22))
