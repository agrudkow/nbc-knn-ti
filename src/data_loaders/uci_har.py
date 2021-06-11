import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame


def get_uci_har(path: str) -> DataFrame:
  df = pd.read_fwf(path, sep=" ", header=None)
  data = df.astype(np.float64).values
  return(data)
