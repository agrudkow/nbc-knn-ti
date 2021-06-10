import numpy as np


def save_data(path: str, data: np.array):
  np.savetxt(path, data, delimiter=',', fmt='%d')