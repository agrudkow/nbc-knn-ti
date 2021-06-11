import numpy as np


def save_data(path: str, data: np.array, fmt: str = '%d'):
  np.savetxt(path, data, delimiter=',', fmt=fmt)