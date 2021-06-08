import unittest
from typing import List, Tuple

import numpy as np
from numpy.core.numeric import Inf

from utils import distance, insort_right

# import numpy as np


class TIkNeighborhoodIndex():

  def __init__(self, data: List[Tuple[float, ...]], dimensions: int, k: int = 3) -> None:
    assert k <= len(data), '`k` is greater than the length of `data`'

    self._k = k
    self._r: Tuple[float, ...] = (0.0,) * dimensions
    self._data = data
    self._dists = self.create_est_dist_list()

  def run(self) -> List[List[int]]:
    knns: List[List[int]] = []

    for i, _ in enumerate(self._dists):
      # _, idx = item

      knns.append(self.ti_k_neighborhood(i))

    return []

  def ti_k_neighborhood(self, p_idx: int) -> List[int]:
    knn: List[Tuple[float, int]] = []
    i = 0
    eps = Inf
    b_idx = f_idx = p_idx

    backwardSearch, b_idx = self.preceding_point(b_idx)
    forwardSearch, f_idx = self.following_point(f_idx)

    knn, b_idx, f_idx, backwardSearch, forwardSearch, i, eps = self.find_first_kcn_fb(
        knn,
        p_idx,
        b_idx,
        f_idx,
        backwardSearch,
        forwardSearch,
        i,
        eps,
    )
    knn, b_idx, backwardSearch, i, eps = self.find_first_kcn_b(
        knn,
        p_idx,
        b_idx,
        backwardSearch,
        i,
        eps,
    )
    knn, f_idx, forwardSearch, i, eps = self.find_first_kcn_f(
        knn,
        p_idx,
        f_idx,
        forwardSearch,
        i,
        eps,
    )

    return []

  def preceding_point(self, idx) -> Tuple[int, bool]:
    if idx > 0:
      return (idx - 1, True)
    else:
      return (idx, False)

  def following_point(self, idx) -> Tuple[int, bool]:
    if idx < (len(self._dists) - 1):
      return (idx + 1, True)
    else:
      return (idx, False)

  def find_first_kcn_fb(
      self,
      knn: List[Tuple[float, int]],
      p_idx: int,
      b_idx: int,
      f_idx: int,
      backwardSearch: bool,
      forwardSearch: bool,
      i: int,
      eps: float,
  ) -> Tuple[List[Tuple[float, int]], int, int, bool, bool, int, float]:
    while backwardSearch and forwardSearch and (i < self._k):
      if (self._dists[p_idx][0] - self._dists[b_idx][0]) < (self._dists[f_idx][0] - self._dists[p_idx][0]):
        dist = distance(self._data[b_idx], self._data[p_idx])
        i += 1
        insort_right(knn, (dist, b_idx), key=lambda x: x[0])
        backwardSearch, b_idx = self.preceding_point(b_idx)
        eps = max(eps, dist)
      else:
        dist = distance(self._data[f_idx], self._data[p_idx])
        i += 1
        insort_right(knn, (dist, f_idx), key=lambda x: x[0])
        forwardSearch, f_idx = self.following_point(f_idx)
        eps = max(eps, dist)

    return (knn, b_idx, f_idx, backwardSearch, forwardSearch, i, eps)

  def find_first_kcn_b(
      self,
      knn: List[Tuple[float, int]],
      p_idx: int,
      b_idx: int,
      backwardSearch: bool,
      i: int,
      eps: float,
  ) -> Tuple[List[Tuple[float, int]], int, bool, int, float]:
    while backwardSearch and (i < self._k):
      dist = distance(self._data[b_idx], self._data[p_idx])
      i += 1
      insort_right(knn, (dist, b_idx), key=lambda x: x[0])
      backwardSearch, b_idx = self.preceding_point(b_idx)
      eps = max(eps, dist)

    return (knn, b_idx, backwardSearch, i, eps)

  def find_first_kcn_f(
      self,
      knn: List[Tuple[float, int]],
      p_idx: int,
      f_idx: int,
      forwardSearch: bool,
      i: int,
      eps: float,
  ) -> Tuple[List[Tuple[float, int]], int, bool, int, float]:
    while forwardSearch and (i < self._k):
      dist = distance(self._data[f_idx], self._data[p_idx])
      i += 1
      insort_right(knn, (dist, f_idx), key=lambda x: x[0])
      forwardSearch, f_idx = self.following_point(f_idx)
      eps = max(eps, dist)

    return (knn, f_idx, forwardSearch, i, eps)

  def verify_k_condidate_neighbours_backward(self):
    pass

  def verify_k_condidate_neighbours_forward(self):
    pass

  def create_est_dist_list(self) -> List[Tuple[float, int]]:
    dist: List[Tuple[float, int]] = []

    for idx, point in enumerate(self._data):
      dist.append((distance(self._r, point), idx))

    return sorted(dist, key=lambda x: x[0])


# -------------------- Test -----------------------------------


class TestTIkNeighborhoodIndex(unittest.TestCase):

  def setUp(self) -> TIkNeighborhoodIndex:
    data: List[Tuple[float, ...]] = [(2., 0.), (2., 2.), (1., 1.)]
    dimensions = 2
    k = 3
    self.c = TIkNeighborhoodIndex(data, dimensions, k)

  def test_create_est_dist_list(self):
    np.testing.assert_almost_equal(self.c.create_est_dist_list(), [(1.4142, 2), (2.0, 0), (2.8284, 1)], 4)

  def test_preceding_point(self):
    self.assertEqual(self.c.preceding_point(0), (0, False), 'Check first point')
    self.assertEqual(self.c.preceding_point(1), (0, True), 'Check second point')

  def test_following_point(self):
    last_point_idx = len(self.c._data) - 1
    self.assertEqual(self.c.following_point(last_point_idx), (last_point_idx, False), 'Check last point')
    self.assertEqual(self.c.following_point(last_point_idx - 1), (last_point_idx, True), 'Check penultimate point')


if __name__ == '__main__':
  unittest.main()
