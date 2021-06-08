import unittest
from typing import Tuple

import numpy as np


def distance(a: Tuple[float, ...], b: Tuple[float, ...]):
  return np.linalg.norm(np.array(a) - np.array(b))


# -------------------- Test -----------------------------------


class TestDistance(unittest.TestCase):

  def test_distance(self):
    r = (0.0,) * 2
    self.assertEqual(distance(r, (0.0, 0.0)), 0.0)
    self.assertEqual(distance(r, (1.0, 0.0)), 1.0)
    self.assertEqual(distance(r, (0.0, 1.0)), 1.0)
    np.testing.assert_almost_equal(distance(r, (1.0, 1.0)), 1.4142, 4)


if __name__ == '__main__':
  unittest.main()
