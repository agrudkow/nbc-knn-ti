import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame


def get_amzn_anon(path: str) -> DataFrame:
  df = pd.read_csv(path, header=0, delimiter=";")
  data = df.astype(np.float64).values
  return(np.delete(data, -1, axis=1))
