from absl import app
from absl import flags
import numpy as np

from nbc import nbc
from utils.save_data import save_data
from utils.read_data import read_data

FLAGS = flags.FLAGS

flags.DEFINE_string('input', default=None, help='Path to input file containing dataset.')
flags.DEFINE_string('output', default=None, help='Path to output file where clastering results will be stored.')
flags.DEFINE_integer('k', default=3, help='Number of nearest neighbours.')
flags.DEFINE_string('index_type',
                    default='ti-kn',
                    help='Type of index which will be used to calculate kNN. Avaliable indices: `ti-kn`, `kn`.')
flags.mark_flag_as_required("input")


def main(_):
  df = read_data(FLAGS.input)
  clusters = nbc(df.values, df.shape[1], FLAGS.k, FLAGS.index_type)

  if FLAGS.output is not None:
    save_data(FLAGS.output, np.array(clusters))

  print(clusters)


if __name__ == '__main__':
  app.run(main)
