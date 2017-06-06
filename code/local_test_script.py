from experiment import ActivationfnExperiment
from graph_drawer import drawexperimentgraph
import tensorflow as tf

if __name__ == "__main__":
  def transSigmoid(down):
    return lambda x : tf.nn.sigmoid(x) - tf.constant(down, dtype=tf.float64)
  expr = ActivationfnExperiment([tf.nn.sigmoid, transSigmoid(0.5)], ["sigmoid", "transsigmoid"], "sigmoid vs transsigmoid", 2, epochs=3000)
  expr.runandsaveoutput()
  drawexperimentgraph(expr, expr.filename)