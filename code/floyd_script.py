from experiment import ActivationfnExperiment
from graph_drawer import drawexperimentgraph
import tensorflow as tf

if __name__ == "__main__":
  def reluVar(a):
    return lambda x : tf.maximum(x*a, tf.constant(0.0, dtype=tf.float64))
  def leakyrelu(x):
    return tf.maximum(x, 0.01*x)
  fns = [leakyrelu, tf.nn.relu]
  tags = ["sigmoid", "relu"]
  expr = ActivationfnExperiment(fns, tags, "test", 2)
  expr.runandsaveoutput()
  drawexperimentgraph(expr, expr.filename)
