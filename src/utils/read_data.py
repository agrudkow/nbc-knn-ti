import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame


def read_data(path: str) -> DataFrame:
  df = pd.read_csv(path, header=None, delimiter=",")
  return (df.astype(np.float64))
