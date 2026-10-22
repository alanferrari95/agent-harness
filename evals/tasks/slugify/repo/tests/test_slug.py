import unittest

from text.slug import slugify


class TestSlugify(unittest.TestCase):
    def test_punctuation(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_extra_spaces(self):
        self.assertEqual(slugify("  Many   spaces "), "many-spaces")
