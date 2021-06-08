from typing import List, Tuple
import unittest

from src.ti_kn_index import TIkNeighborhoodIndex


class TestTIkNeighborhoodIndex(unittest.TestCase):

  def __init_taste_case(self) -> TIkNeighborhoodIndex:
    data: List[Tuple[float, ...]] = [(2., 0.), (2., 2.), (1., 1.)]
    dimensions = 3
    k = 3
    c = TIkNeighborhoodIndex(data, dimensions, k)

    return c

  def test_following_point(self):
    c = self.__init_taste_case()

    self.assertAlmostEqual(c.create_est_dist_list(), [])


if __name__ == '__main__':
  unittest.main()

