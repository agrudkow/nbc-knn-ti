import unittest
from typing import List, Tuple

import numpy as np

from utils import distance, insort_right

# import numpy as np


class TIkNeighborhoodIndex():

  def __init__(self, data: List[Tuple[float, ...]], dimensions: int, k: int = 3) -> None:
    assert k <= len(data), '`k` is greater than the length of `data`'

    self._k = k
    self._r: Tuple[float, ...] = (0.0,) * dimensions
    self._data = data
    self._dists = self.create_est_dist_list()

  def run(self) -> List[List[Tuple[float, int]]]:
    knns: List[List[Tuple[float, int]]] = []

    for i, _ in enumerate(self._dists):
      # _, idx = item

      knns.append(self.ti_k_neighborhood(i))

    return knns

  def ti_k_neighborhood(self, p_idx: int) -> List[Tuple[float, ...]]:
    knn: List[Tuple[float, int]] = []
    i = 0
    eps = 0
    b_idx = f_idx = p_idx

    b_idx, backwardSearch = self.preceding_point(b_idx)
    f_idx, forwardSearch = self.following_point(f_idx)

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
    knn, b_idx, backwardSearch, eps = self.verify_k_condidate_neighbours_backward(
        knn,
        p_idx,
        b_idx,
        backwardSearch,
        eps,
    )
    knn, f_idx, forwardSearch, eps = self.verify_k_condidate_neighbours_forward(
        knn,
        p_idx,
        f_idx,
        forwardSearch,
        eps,
    )

    return knn

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
        dist = self.calc_real_distance(b_idx, p_idx)
        i += 1
        insort_right(knn, (dist, b_idx), key=lambda x: x[0])
        b_idx, backwardSearch = self.preceding_point(b_idx)
        eps = max(eps, dist)
      else:
        dist = self.calc_real_distance(f_idx, p_idx)
        i += 1
        insort_right(knn, (dist, f_idx), key=lambda x: x[0])
        f_idx, forwardSearch = self.following_point(f_idx)
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
      dist = self.calc_real_distance(b_idx, p_idx)
      i += 1
      insort_right(knn, (dist, b_idx), key=lambda x: x[0])
      b_idx, backwardSearch = self.preceding_point(b_idx)
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
      dist = self.calc_real_distance(f_idx, p_idx)
      i += 1
      insort_right(knn, (dist, f_idx), key=lambda x: x[0])
      f_idx, forwardSearch = self.following_point(f_idx)
      eps = max(knn, key=lambda e: e[0])

    return (knn, f_idx, forwardSearch, i, eps)

  def verify_k_condidate_neighbours_backward(self, knn: List[Tuple[float,
                                                                   int]], p_idx: int, b_idx: int, backwardSearch: bool,
                                             eps: float) -> Tuple[List[Tuple[float, int]], int, bool, float]:
    while backwardSearch and (self._dists[p_idx][0] - self._dists[b_idx][0]) <= eps:
      dist = self.calc_real_distance(b_idx, p_idx)
      if dist < eps:
        i_list = [n for n in knn if n[0] == eps]
        i = len(i_list)
        if (len(knn) - i) >= (self._k - 1):
          knn = [e for e in knn if e not in i_list]
          insort_right(knn, (dist, b_idx))
          eps = max(eps, dist)
        else:
          insort_right(knn, (dist, b_idx))
      elif dist == eps:
        insort_right(knn, (dist, b_idx))

      b_idx, backwardSearch = self.preceding_point(b_idx)

    return (knn, b_idx, backwardSearch, eps)

  def verify_k_condidate_neighbours_forward(self, knn: List[Tuple[float,
                                                                  int]], p_idx: int, f_idx: int, forwardSearch: bool,
                                            eps: float) -> Tuple[List[Tuple[float, int]], int, bool, float]:
    while forwardSearch and (self._dists[f_idx][0] - self._dists[p_idx][0]) <= eps:
      dist = self.calc_real_distance(f_idx, p_idx)
      if dist < eps:
        i_list = [n for n in knn if n[0] == eps]
        i = len(i_list)
        if (len(knn) - i) >= (self._k - 1):
          knn = [e for e in knn if e not in i_list]
          insort_right(knn, (dist, f_idx))
          eps = max(eps, dist)
        else:
          insort_right(knn, (dist, f_idx))
      elif dist == eps:
        insort_right(knn, (dist, f_idx))

      f_idx, forwardSearch = self.preceding_point(f_idx)

    return (knn, f_idx, forwardSearch, eps)

  def get_idx_from_dist(self, idx) -> int:
    return self._dists[idx][1]

  def calc_real_distance(self, idx_1, idx_2) -> float:
    return distance(self._data[self.get_idx_from_dist(idx_1)], self._data[self.get_idx_from_dist(idx_2)])

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

  def test_get_idx_from_dist(self):
    self.assertEqual(self.c.get_idx_from_dist(0), 2, 'Check index translation from dists to data')

  def test_preceding_point(self):
    self.assertEqual(self.c.preceding_point(0), (0, False), 'Check first point')
    self.assertEqual(self.c.preceding_point(1), (0, True), 'Check second point')

  def test_following_point(self):
    last_point_idx = len(self.c._data) - 1
    self.assertEqual(self.c.following_point(last_point_idx), (last_point_idx, False), 'Check last point')
    self.assertEqual(self.c.following_point(last_point_idx - 1), (last_point_idx, True), 'Check penultimate point')

  def test_find_first_kcn_f(self):
    knn = []
    f_idx = 1
    p_idx = 1
    i = 0
    eps = 0
    f_idx, forwardSearch = self.c.following_point(f_idx)
    test_knn, test_f_idx, test_forwardSearch, test_i, test_eps = self.c.find_first_kcn_f(
        knn, p_idx, f_idx, forwardSearch, i, eps)
    np.testing.assert_almost_equal(test_knn, [(2., 2)], 4, 'Test knn')
    self.assertEqual(test_f_idx, 2, 'Test f_idx')
    self.assertEqual(test_forwardSearch, False, 'Test forwardSearch')
    self.assertEqual(test_i, 1, 'Test i')
    self.assertAlmostEqual(test_eps, 2, 4, 'Test eps')

  def test_find_first_kcn_b(self):
    knn = []
    b_idx = 1
    p_idx = 1
    i = 0
    eps = 0
    b_idx, backwardSearch = self.c.preceding_point(b_idx)
    test_knn, test_b_idx, test_backwardSearch, test_i, test_eps = self.c.find_first_kcn_b(
        knn, p_idx, b_idx, backwardSearch, i, eps)
    np.testing.assert_almost_equal(test_knn, [(1.4142, 0)], 4, 'Test knn')
    self.assertEqual(test_b_idx, 0, 'Test b_idx')
    self.assertEqual(test_backwardSearch, False, 'Test backwardSearch')
    self.assertEqual(test_i, 1, 'Test i')
    self.assertAlmostEqual(test_eps, 1.4142, 4, 'Test eps')

  def test_find_first_kcn_bf(self):
    knn = []
    b_idx = 1
    f_idx = 1
    p_idx = 1
    i = 0
    eps = 0
    b_idx, backwardSearch = self.c.preceding_point(b_idx)
    f_idx, forwardSearch = self.c.following_point(f_idx)
    (test_knn, test_b_idx, test_f_idx, test_backwardSearch, test_forwardSearch, test_i,
     test_eps) = self.c.find_first_kcn_fb(knn, p_idx, b_idx, f_idx, backwardSearch, forwardSearch, i, eps)
    np.testing.assert_almost_equal(test_knn, [(1.4142, 0)], 4, 'Test knn')
    self.assertEqual(test_b_idx, 0, 'Test b_idx')
    self.assertEqual(test_f_idx, 2, 'Test f_idx')
    self.assertEqual(test_backwardSearch, False, 'Test backwardSearch')
    self.assertEqual(test_forwardSearch, True, 'Test forwardSearch')
    self.assertEqual(test_i, 1, 'Test i')
    self.assertAlmostEqual(test_eps, 1.4142, 4, 'Test eps')


if __name__ == '__main__':
  # unittest.main()
  data: List[Tuple[float, ...]] = [(2., 0.), (2., 2.), (1., 1.)]
  dimensions = 2
  k = 3
  c = TIkNeighborhoodIndex(data, dimensions, k)
  knns = c.run()

  print(knns)
