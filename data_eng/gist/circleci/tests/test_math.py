import unittest
from src.math import Math


class MathTest(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(Math.addition(3, 4), 8)


if __name__ == '__main__':
    unittest.main()
