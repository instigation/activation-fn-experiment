from experiment import ActivationfnExperiment
import tensorflow as tf

if __name__ == "__main__":
  def reluVar(a):
    return lambda x : tf.maximum(x*a, tf.constant(0.0, dtype=tf.float64))
  fns = [reluVar(1+0.1*i) for i in range(10)]
  tags = ["gradient"+str(1+0.1*i) for i in range(10)]
  expr = ActivationfnExperiment(fns, tags, "relu with different gradients", 20)
  expr.runandsaveoutput()