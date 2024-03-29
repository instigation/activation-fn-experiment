from experiment import ActivationfnExperiment
import tensorflow as tf

if __name__ == "__main__":
  floyd = False
  def transSigmoid(down):
    return lambda x : tf.nn.sigmoid(x) - tf.constant(down, dtype=tf.float64)
  def strSigmoid(ratio):
    return lambda x : tf.nn.sigmoid(x) * ratio
  fns = [strSigmoid(1 + 0.2*i) for i in range(10)]
  tags = ["sigmoid"+str(1 + 0.2*i) for i in range(10)]
  expr = ActivationfnExperiment(fns, tags, "strsigmoid", 10)
  if floyd:
    expr.runandsaveoutput()
  else:
    from code.graph_drawer import drawexperimentgraph
    expr.loadoutput()
    drawexperimentgraph(expr, expr.filename)