import unittest

from insort_right import insort_right


class TestInsortRight(unittest.TestCase):

  def test_insort_right(self):
    data = [(1., 0), (2., 1), (3., 2)]
    insort_right(data, (1., 100), key=lambda x: x[0])
    self.assertEqual(data, [(1., 0), (1., 100), (2., 1), (3., 2)], 'Check insert of existing key')

    data = [(2., 1), (3., 2)]
    insort_right(data, (1., 0), key=lambda x: x[0])
    self.assertEqual(data, [(1., 0), (2., 1), (3., 2)], 'Check insert of into the first position')

    data = [(1., 0), (2., 1)]
    insort_right(data, (3., 2), key=lambda x: x[0])
    self.assertEqual(data, [(1., 0), (2., 1), (3., 2)], 'Check insert of into the last position')

    data = []
    insort_right(data, (3., 2), key=lambda x: x[0])
    self.assertEqual(data, [(3., 2)], 'Check insert of into the empty list')


if __name__ == '__main__':
  unittest.main()
