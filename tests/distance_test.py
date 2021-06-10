import unittest

from utils.distance import distance


class TestDistance(unittest.TestCase):

  def test_distance(self):
    r = (0.0,) * 2
    self.assertEqual(distance(r, (0.0, 0.0)), 0.0)
    self.assertEqual(distance(r, (1.0, 0.0)), 1.0)
    self.assertEqual(distance(r, (0.0, 1.0)), 1.0)
    self.assertAlmostEqual(distance(r, (1.0, 1.0)), 1.41421356)


if __name__ == '__main__':
  unittest.main()
