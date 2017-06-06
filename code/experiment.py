import network as network
import tensorflow as tf
from data import Data
import numpy as np


def nametofilename(str):
  return str.replace(" ", "_")


def average(arrays):
  n = len(arrays)
  sum = np.sum(arrays, axis=0)
  return [elem / n for elem in sum]


class ExperimentManager(object):
  def __init__(self, experiments):
    self.experiments = experiments


class ActivationfnExperiment(object):
  def __init__(self, activationfns, tags, name, n_iteration, epochs=20000):
    self.activationfns = activationfns
    self.tags = tags
    self.name = name
    self.filename = nametofilename(name)
    self.data = Data()
    self.n_iteration = n_iteration
    self.epochs = epochs

  def runone(self, activationfn):
    with tf.Graph().as_default():
      dtype = tf.float64
      x = tf.placeholder(dtype, [None, 784])
      layer01 = network.FullyConnectedLayer(784, 100, activationfn, x, dtype)
      layer02 = network.FullyConnectedLayer(100, 10, tf.nn.softmax, layer01.var_out, dtype)

      def cross_entropy(y, y_):
        return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

      net = network.Network([layer01, layer02], cross_entropy)
      mnist = self.data.loadmnistdata()
      ret = []
      for _ in range(self.n_iteration):
        ret.append(net.runmnist(100, self.epochs, 0.5, mnist))
      self.raw_output.append(ret)
      return average(ret)

  def run(self):
    self.raw_output = []
    self.output = []
    for activationfn in self.activationfns:
      self.output.append(self.runone(activationfn))
    return self.output

  def runandsaveoutput(self):
    self.run()
    self.data.savetxt(self.filename, self.output)

  def loadoutput(self):
    self.output = self.data.loadoutputtxt(self.filename)


if __name__ == "__main__":
  from code.graph_drawer import drawexperimentgraph

  onfloyd = True
  if onfloyd:
    expr = ActivationfnExperiment([tf.nn.sigmoid, tf.nn.relu], ["sigmoid", "relu"], "sigmoid vs relu", 2)
    expr.runandsaveoutput()
    drawexperimentgraph(expr, expr.filename)
  else:
    empty_expr = ActivationfnExperiment([], ["sigmoid", "relu"], "sigmoid vs relu", 2)
    empty_expr.loadoutput()
    drawexperimentgraph(empty_expr, empty_expr.filename)
