from experiment import ActivationfnExperiment
import tensorflow as tf

if __name__ == "__main__":
  def resigmoid(t):
    return lambda x: tf.maximum(x / 20 - t, tf.minimum(x / 20 + t, x / 0.8))


  def linear():
    return lambda x: x


  def triple(a, t):
    return lambda x: tf.maximum(tf.minimum(x, tf.constant(t, dtype=tf.float64) + a * x),
                                tf.constant(0, dtype=tf.float64))


  additional = True
  if not additional:
    fns = []
    tags = []
    for a in range(5):
      for t in range(5):
        fns.append(triple(0.6 + 0.2 * a, 2 ** t))
        tags.append("%.1f,%d" % (0.6 + 0.2 * a, 2 ** t))
    fns.append(tf.nn.relu)
    tags.append("relu")
    expr = ActivationfnExperiment(fns, tags, "triple vs relu", 10)
    floyd = False
    if floyd:
      expr.runandsaveoutput()
    else:
      from graph_drawer import drawexperimentgraph
      expr.loadoutput()
      drawexperimentgraph(expr, expr.filename)

  else:
    fns = []
    tags = []
    for t in range(14, 19):
      fns.append(triple(0.8, t))
      tags.append("%.1f,%d" % (0.8, t))
    fns.append(tf.nn.relu)
    tags.append("relu")
    expr = ActivationfnExperiment(fns, tags, "detailed triple", 10)
    floyd = False
    if floyd:
      expr.runandsaveoutput()
    else:
      from graph_drawer import drawexperimentgraph
      expr.loadoutput()
      drawexperimentgraph(expr, expr.filename)

