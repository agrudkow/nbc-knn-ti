from typing import Tuple

import numpy as np

from utils import distance, insort_right, timing
from utils.types import KNN, KNNS, R_KNNS


class TIkNeighborhoodIndex():

  def __init__(self, data: np.array, dimensions: int, k: int = 3) -> None:
    assert k <= len(data), '`k` is greater than the length of `data`'

    self._k = k
    self._r: np.array = np.zeros(dimensions)
    self._data = data
    self._dists = self.create_est_dist_list()

  @timing
  def run(self, ):
    knns: KNNS = [list() for _ in range(len(self._data))]
    r_knns: R_KNNS = [list() for _ in range(len(self._data))]

    for i, item in enumerate(self._dists):
      _, idx = item
      knns[idx] = self.ti_k_neighborhood(i)
      for n in knns[idx]:
        r_knns[n[1]].append(idx)

    return knns, r_knns

  def ti_k_neighborhood(self, p_idx: int) -> np.array:
    knn: KNN = []
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
      knn: KNN,
      p_idx: int,
      b_idx: int,
      f_idx: int,
      backwardSearch: bool,
      forwardSearch: bool,
      i: int,
      eps: float,
  ) -> Tuple[KNN, int, int, bool, bool, int, float]:
    while backwardSearch and forwardSearch and (i < self._k):
      if (self._dists[p_idx][0] - self._dists[b_idx][0]) < (self._dists[f_idx][0] - self._dists[p_idx][0]):
        dist = self.calc_real_distance(b_idx, p_idx)
        i += 1
        insort_right(knn, (dist, self.get_idx_from_dist(b_idx)), key=lambda x: x[0])
        b_idx, backwardSearch = self.preceding_point(b_idx)
        eps = max(eps, dist)
      else:
        dist = self.calc_real_distance(f_idx, p_idx)
        i += 1
        insort_right(knn, (dist, self.get_idx_from_dist(f_idx)), key=lambda x: x[0])
        f_idx, forwardSearch = self.following_point(f_idx)
        eps = max(eps, dist)

    return (knn, b_idx, f_idx, backwardSearch, forwardSearch, i, eps)

  def find_first_kcn_b(
      self,
      knn: KNN,
      p_idx: int,
      b_idx: int,
      backwardSearch: bool,
      i: int,
      eps: float,
  ) -> Tuple[KNN, int, bool, int, float]:
    while backwardSearch and (i < self._k):
      dist = self.calc_real_distance(b_idx, p_idx)
      i += 1
      insort_right(knn, (dist, self.get_idx_from_dist(b_idx)), key=lambda x: x[0])
      b_idx, backwardSearch = self.preceding_point(b_idx)
      eps = max(eps, dist)

    return (knn, b_idx, backwardSearch, i, eps)

  def find_first_kcn_f(
      self,
      knn: KNN,
      p_idx: int,
      f_idx: int,
      forwardSearch: bool,
      i: int,
      eps: float,
  ) -> Tuple[KNN, int, bool, int, float]:
    while forwardSearch and (i < self._k):
      dist = self.calc_real_distance(f_idx, p_idx)
      i += 1
      insort_right(knn, (dist, self.get_idx_from_dist(f_idx)), key=lambda x: x[0])
      f_idx, forwardSearch = self.following_point(f_idx)
      eps = max(eps, dist)

    return (knn, f_idx, forwardSearch, i, eps)

  def verify_k_condidate_neighbours_backward(self, knn: KNN, p_idx: int, b_idx: int, backwardSearch: bool,
                                             eps: float) -> Tuple[KNN, int, bool, float]:
    while backwardSearch and (self._dists[p_idx][0] - self._dists[b_idx][0]) <= eps:
      dist = self.calc_real_distance(b_idx, p_idx)
      if dist < eps:
        i_list = [n for n in knn if n[0] == eps]
        i = len(i_list)
        if (len(knn) - i) >= (self._k - 1):
          knn = [e for e in knn if e not in i_list]
          insort_right(knn, (dist, self.get_idx_from_dist(b_idx)), key=lambda x: x[0])
          eps = max(eps, dist)
        else:
          insort_right(knn, (dist, self.get_idx_from_dist(b_idx)), key=lambda x: x[0])
      elif dist == eps:
        insort_right(knn, (dist, self.get_idx_from_dist(b_idx)), key=lambda x: x[0])

      b_idx, backwardSearch = self.preceding_point(b_idx)

    return (knn, b_idx, backwardSearch, eps)

  def verify_k_condidate_neighbours_forward(self, knn: KNN, p_idx: int, f_idx: int, forwardSearch: bool,
                                            eps: float) -> Tuple[KNN, int, bool, float]:
    while (forwardSearch and (self._dists[f_idx][0] - self._dists[p_idx][0]) <= eps):
      dist = self.calc_real_distance(f_idx, p_idx)
      if dist < eps:
        i_list = [n for n in knn if n[0] == eps]
        i = len(i_list)
        if (len(knn) - i) >= (self._k - 1):
          knn = [e for e in knn if e not in i_list]
          insort_right(knn, (dist, self.get_idx_from_dist(f_idx)), key=lambda x: x[0])
          eps = max(eps, dist)
        else:
          insort_right(knn, (dist, self.get_idx_from_dist(f_idx)), key=lambda x: x[0])
      elif dist == eps:
        insort_right(knn, (dist, self.get_idx_from_dist(f_idx)), key=lambda x: x[0])

      f_idx, forwardSearch = self.following_point(f_idx)

    return (knn, f_idx, forwardSearch, eps)

  def get_idx_from_dist(self, idx) -> int:
    return self._dists[idx][1]

  def calc_real_distance(self, idx_1, idx_2) -> float:
    return distance(self._data[self.get_idx_from_dist(idx_1)], self._data[self.get_idx_from_dist(idx_2)])

  def create_est_dist_list(self) -> KNN:
    dist: KNN = []

    for idx, point in enumerate(self._data):
      dist.append((distance(self._r, point), idx))

    return sorted(dist, key=lambda x: x[0])


if __name__ == '__main__':
  data: np.array = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [1., 0.], [0., 1.]])
  dimensions = 2
  k = 3
  c = TIkNeighborhoodIndex(data, dimensions, k)
  knns, r_knns = c.run()

  print(knns[0])
  print(r_knns[0])
