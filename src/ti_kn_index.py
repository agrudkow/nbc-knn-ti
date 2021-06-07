from typing import List, Tuple

# import numpy as np


class TIkNeighborhoodIndex():

  def __init__(self,
               data: List[Tuple[float, ...]],
               dimensions: int,
               k: int = 3) -> None:
    self._k = k
    self._data = data

  def run(self) -> List[List[int]]:
    knn: List[List[int]] = []
    dist: List[Tuple[float, ...]] = []
    for point in self._data:
      # calc distanc from r
      pass

    for i, point in enumerate(sorted(dist)):
      knn.append(self._ti_k_neighborhood(point))

    return []

  def _ti_k_neighborhood(self, poitnt_id: Tuple[float, ...]) -> List[int]:
    pass

  def _preceding_point(self):
    pass

  def _following_point(self):
    pass

  def find_first_k_condidate_neighbours_forward_and_backward(self):
    pass

  def verify_k_condidate_neighbours_backward(self):
    pass

  def verify_k_condidate_neighbours_forward(self):
    pass
