import unittest

import numpy as np

from ti_kn_index import TIkNeighborhoodIndex


class TestTIkNeighborhoodIndex(unittest.TestCase):

  def setUp(self) -> TIkNeighborhoodIndex:
    data: np.array = np.array([[2., 0.], [2., 2.], [1., 1.]])
    dimensions = 2
    k = 3
    self.c = TIkNeighborhoodIndex(data, dimensions, k)

  def test_create_est_dist_list(self):
    np.testing.assert_almost_equal(self.c.create_est_dist_list(), [[1.4142, 2], [2.0, 0], [2.8284, 1]], 4)

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
    np.testing.assert_almost_equal(test_knn, [(2., 1)], 4, 'Test knn')
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
    np.testing.assert_almost_equal(test_knn, [(1.4142, 2)], 4, 'Test knn')
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
    np.testing.assert_almost_equal(test_knn, [(1.4142, 2)], 4, 'Test knn')
    self.assertEqual(test_b_idx, 0, 'Test b_idx')
    self.assertEqual(test_f_idx, 2, 'Test f_idx')
    self.assertEqual(test_backwardSearch, False, 'Test backwardSearch')
    self.assertEqual(test_forwardSearch, True, 'Test forwardSearch')
    self.assertEqual(test_i, 1, 'Test i')
    self.assertAlmostEqual(test_eps, 1.4142, 4, 'Test eps')


if __name__ == '__main__':
  unittest.main()
