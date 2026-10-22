import unittest

from settings.parser import parse_config


class TestParser(unittest.TestCase):
    def test_strips_whitespace(self):
        text = "name = demo\n\nport= 8080 \n"
        self.assertEqual(parse_config(text), {"name": "demo", "port": "8080"})

    def test_indented_comment(self):
        self.assertEqual(parse_config("  # note\nmode=fast"), {"mode": "fast"})
