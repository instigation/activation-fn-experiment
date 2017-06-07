from experiment import ActivationfnExperiment
from graph_drawer import drawexperimentgraph
import tensorflow as tf

if __name__ == "__main__":
  def reluVar(a):
    return lambda x : tf.maximum(x*a, tf.constant(0.0, dtype=tf.float64))
  fns = [reluVar(1-0.05*i) for i in range(10)]
  tags = ["gradient"+str(1-0.05*i) for i in range(10)]
  expr = ActivationfnExperiment(fns, tags, "relu with smaller gradients", 10)
  expr.runandsaveoutput()
  drawexperimentgraph(expr, expr.filename)
