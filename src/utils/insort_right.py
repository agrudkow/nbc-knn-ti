from typing import Callable, List, TypeVar
from bisect import bisect_right

T = TypeVar('T')


def insort_right(data: List[T], item: T, key: Callable[[str], int]):
  keys = [key(r) for r in data]
  data.insert(bisect_right(keys, key(item)), item)
