import unittest

import numpy as np

from nbc import nbc


class TestNBC(unittest.TestCase):

  def test_nbc(self):
    # given
    cluster_0 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    # Cluster 1 moved "far" from Cluster 0
    cluster_1 = cluster_0 + 10
    k = 2

    # when
    c = nbc(data=np.vstack((cluster_0, cluster_1)), dimensions=len(cluster_0[0]), k=k)

    # assert
    expected_clusters = [0, 0, 0, 1, 1, 1]
    self.assertListEqual(c, expected_clusters)

  def test_ti_nbc(self):
    # given
    cluster_0 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    # Cluster 1 moved "far" from Cluster 0
    cluster_1 = cluster_0 + 10
    k = 2

    # when
    c = nbc(data=np.vstack((cluster_0, cluster_1)), dimensions=len(cluster_0[0]), k=k, ti_index=True)

    # assert
    expected_clusters = [0, 0, 0, 1, 1, 1]
    self.assertListEqual(c, expected_clusters)


if __name__ == '__main__':
  unittest.main()
