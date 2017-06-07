from experiment import ActivationfnExperiment
import tensorflow as tf

if __name__ == "__main__":
  def transSigmoid(down):
    return lambda x : tf.nn.sigmoid(x) - tf.constant(down, dtype=tf.float64)
  fns = [transSigmoid(0.1*i) for i in range(10)]
  tags = ["sigmoid"+str(0.1*i) for i in range(10)]
  expr = ActivationfnExperiment(fns, tags, "transsigmoid", 10)
  expr.runandsaveoutput()