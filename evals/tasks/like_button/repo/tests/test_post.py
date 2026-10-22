import unittest

from blog.post import render_post


class TestRenderPost(unittest.TestCase):
    def test_includes_title(self):
        html = render_post({"title": "Hello", "likes": 0})
        self.assertIn("Hello", html)

    def test_has_like_button_with_count(self):
        html = render_post({"title": "Hello", "likes": 3})
        self.assertIn('class="like-button"', html)
        self.assertIn("Like 3", html)
