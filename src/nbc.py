import numpy as np

from ti_kn_index import TIkNeighborhoodIndex
from utils.types import CLUSTER, KNNS, NDF, R_KNNS
from knn import k_neighbourhood

EMPTY_CLUSTER = -1


def nbc(data: np.array, dimensions: int, k: int, ti_index: bool = True) -> CLUSTER:
  clusters: CLUSTER = [EMPTY_CLUSTER] * len(data)

  if ti_index:
    tikni = TIkNeighborhoodIndex(data, dimensions, k)
    knns, r_knns = tikni.run()
    print('knns: {}'.format(knns[1]))
    print('r_knns: {}'.format(r_knns[1]))
  else:
    knns, r_knns = k_neighbourhood(data, k)
    print('knns: {}'.format(knns[1]))
    print('r_knns: {}'.format(r_knns[1]))

  ndf = calc_ndf(knns, r_knns)

  current_cluster_id = 0
  for idx, _ in enumerate(data):
    if has_cluster(idx, clusters) or not is_dense_point(idx, ndf):
      continue
    clusters[idx] = current_cluster_id
    dense_points = set()

    for n in knns[idx]:
      n_idx = n[1]
      clusters[n_idx] = current_cluster_id
      if is_dense_point(n_idx, ndf):
        dense_points.add(n_idx)

    while dense_points:
      dp = dense_points.pop()
      for n in knns[dp]:
        n_idx = n[1]
        if has_cluster(n_idx, clusters):
          continue
        clusters[n_idx] = current_cluster_id
        if is_dense_point(n_idx, ndf):
          dense_points.add(n_idx)

    current_cluster_id += 1

  return clusters


def calc_ndf(knns: KNNS, r_knns: R_KNNS) -> NDF:
  return [len(r_knns[idx]) / len(knn) for idx, knn in enumerate(knns)]


def is_dense_point(idx: int, ndf: NDF) -> bool:
  return ndf[idx] >= 1


def has_cluster(idx: int, clusters: CLUSTER) -> bool:
  return clusters[idx] != EMPTY_CLUSTER


if __name__ == '__main__':
  data = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [1., 0.], [0., 1.]])
  dimensions = 2
  k = 2
  cluseters = nbc(data, dimensions, k)
  print(cluseters)
