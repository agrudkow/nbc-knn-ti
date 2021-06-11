from typing import List, Tuple
import numpy as np

from utils import distance
from utils.types import KNNS, R_KNNS
from utils.timing import timing


@timing
def k_neighbourhood(data: np.ndarray, k: int) -> Tuple[KNNS, R_KNNS]:
  knns: List[List[Tuple[float, int]]] = [list() for _ in range(len(data))]
  r_knns: List[List[int]] = [list() for _ in range(len(data))]

  for idx1, v1 in enumerate(data):
    neighbour_candidates = []
    for idx2, v2 in enumerate(data):
      if idx1 != idx2:
        dist = distance(v1, v2)
        neighbour_candidates.append((dist, idx2))
    neighbour_candidates.sort(key=lambda t: t[0])
    eps = neighbour_candidates[:k][-1][0]

    neighbours = []
    for nc in neighbour_candidates:
      if nc[0] > eps:
        break
      neighbours.append(nc)

    knns[idx1] = neighbours
    for nc in knns[idx1]:
      r_knns[nc[1]].append(idx1)

  return knns, r_knns


if __name__ == '__main__':
  data: np.array = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [1., 0.], [0., 1.]])

  knns, r_knns = k_neighbourhood(data, 3)

  print(knns[0])
  print(r_knns[0])
