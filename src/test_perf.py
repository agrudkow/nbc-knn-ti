from time import time, strftime
from absl import app
from absl import flags

import numpy as np

from data_loaders.absenteeims import get_absenteeims
from data_loaders.uci_har import get_uci_har
from knn import k_neighbourhood
from utils.save_data import save_data
from ti_kn_index import TIkNeighborhoodIndex

FLAGS = flags.FLAGS

flags.DEFINE_string('test_type',
                    default='absenteeims',
                    help='Type of test to run. Avaliable option: `absenteeims`, `uci_har`.')
flags.DEFINE_string('index_type',
                    default='ti-kn',
                    help='Type of index which will be used to calculate kNN. Avaliable indices: `ti-kn`, `kn`.')


def run_absenteeims():
  data = get_absenteeims('data/absenteeism/Absenteeism_at_work.csv')
  dims = data.shape[1]

  if FLAGS.index_type == 'ti-kn':
    results_ti_kn = []
    for k in [10]:
      t_start = time()
      tikni = TIkNeighborhoodIndex(data, dims, k)
      tikni.run()
      t_end = time()
      eval_time = t_end - t_start
      results_ti_kn.append([k, eval_time])
      print('[TI_KN] k: {} time: {}'.format(k, eval_time))

    save_data('results/perf_test/absenteeism_ti_kn_{}.csv'.format(strftime("%Y_%m_%d_%H_%M")),
              np.array(results_ti_kn),
              fmt='%f')
  else:
    results_kn = []
    for k in [10]:
      t_start = time()
      k_neighbourhood(data, k)
      t_end = time()
      eval_time = t_end - t_start
      results_kn.append([k, eval_time])
      print('[KN] k: {} time: {}'.format(k, eval_time))

    save_data('results/perf_test/absenteeism_kn_{}.csv'.format(strftime("%Y_%m_%d_%H_%M")),
              np.array(results_kn),
              fmt='%f')


def run_uci_har():
  data = get_uci_har('data/uci_har/UCI_HAR_Dataset/train/X_train.txt')

  dims = data.shape[1]

  if FLAGS.index_type == 'ti-kn':
    results_ti_kn = []
    for k in [10]:
      t_start = time()
      tikni = TIkNeighborhoodIndex(data, dims, k)
      tikni.run()
      t_end = time()
      eval_time = t_end - t_start
      results_ti_kn.append([k, eval_time])
      print('[TI_KN] k: {} time: {}'.format(k, eval_time))

    save_data('results/perf_test/uci_har_ti_kn_{}.csv'.format(strftime("%Y_%m_%d_%H_%M")),
              np.array(results_ti_kn),
              fmt='%f')
  else:
    results_kn = []
    for k in [10]:
      t_start = time()
      k_neighbourhood(data, k)
      t_end = time()
      eval_time = t_end - t_start
      results_kn.append([k, eval_time])
      print('[KN] k: {} time: {}'.format(k, eval_time))

    save_data('results/perf_test/uci_har_kn_{}.csv'.format(strftime("%Y_%m_%d_%H_%M")), np.array(results_kn), fmt='%f')


def run_tests(_):
  if FLAGS.test_type == 'absenteeims':
    run_absenteeims()
  elif FLAGS.test_type == 'uci_har':
    run_uci_har()
  else:
    raise NotImplementedError('Test type `{}` is not implemented.'.format(FLAGS.test_type))


if __name__ == '__main__':
  app.run(run_tests)
