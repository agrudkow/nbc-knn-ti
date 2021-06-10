import unittest

import numpy as np

from distance import distance


class TestDistance(unittest.TestCase):

  def test_distance(self):
    r = np.zeros(2)
    self.assertEqual(distance(r, np.array((0.0, 0.0))), 0.0)
    self.assertEqual(distance(r, np.array((1.0, 0.0))), 1.0)
    self.assertEqual(distance(r, np.array((0.0, 1.0))), 1.0)
    np.testing.assert_almost_equal(distance(r, np.array((1.0, 1.0))), 1.4142, 4)


if __name__ == '__main__':
  unittest.main()
