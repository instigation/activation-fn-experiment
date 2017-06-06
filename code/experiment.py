import code.network as network
import tensorflow as tf
from code.data import Data


def nametofilename(str):
  return str.replace(" ", "_")


class ExperimentManager(object):
  def __init__(self, experiments):
    self.experiments = experiments


class ActivationfnExperiment(object):
  def __init__(self, activationfns, tags, name):
    self.activationfns = activationfns
    self.tags = tags
    self.name = name
    self.filename = nametofilename(name)
    self.data = Data()

  def runone(self, activationfn):
    dtype = tf.float64
    x = tf.placeholder(dtype, [None, 784])
    layer01 = network.FullyConnectedLayer(784, 100, activationfn, x, dtype)
    layer02 = network.FullyConnectedLayer(100, 10, tf.nn.softmax, layer01.var_out, dtype)

    def cross_entropy(y, y_):
      return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    net = network.Network([layer01, layer02], cross_entropy)
    mnist = self.data.loadmnistdata()
    return net.runmnist(100, 20000, 0.5, mnist)

  def run(self):
    self.output = []
    for activationfn in self.activationfns:
      self.output.append(self.runone(activationfn))
    return self.output

  def runandsaveoutput(self):
    self.run()
    self.data.savetxt(self.filename, self.output)


if __name__ == "__main__":
  expr = ActivationfnExperiment([tf.nn.sigmoid, tf.nn.relu], ["sigmoid", "relu"], "sigmoid vs relu")
  expr.run()

  from code.graph_drawer import drawexperimentgraph

  drawexperimentgraph(expr, expr.filename)
