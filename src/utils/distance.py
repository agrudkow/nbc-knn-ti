import numpy as np


def distance(a: np.array, b: np.array) -> float:
  return np.linalg.norm(a - b).item()
