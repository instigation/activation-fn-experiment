from experiment import ActivationfnExperiment
import tensorflow as tf
from graph_drawer import drawexperimentgraph

if __name__ == "__main__":
  def resigmoid(t):
    return lambda x: tf.maximum(x / 20 - t, tf.minimum(x / 20 + t, x / 0.8))


  def linear():
    return lambda x: x


  def triple(a, t):
    return lambda x: tf.maximum(tf.minimum(x, tf.constant(t, dtype=tf.float64) + a * x),
                                tf.constant(0, dtype=tf.float64))

  fns = [tf.nn.relu, TODO, tf.nn.tanh, tf.nn.sigmoid]
  tags = ["relu", "resigmoid", "tanh", "sigmoid"]
  expr = ActivationfnExperiment(fns, tags, "resigmoid", 10)
  expr.runandsaveoutput()
  drawexperimentgraph(expr, expr.filename)