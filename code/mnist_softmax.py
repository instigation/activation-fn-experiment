import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape, dtype=tf.float64):
    initial = tf.truncated_normal(shape, stddev=0.01, dtype=tf.float64)
    return tf.Variable(initial, dtype=dtype)

def bias_variable(shape, dtype=tf.float64):
    initial = tf.constant(0.0, shape=shape, dtype=tf.float64)
    return tf.Variable(initial, dtype=dtype)

def loadData():
  ## load data
  mnist = input_data.read_data_sets("../input/MNIST_data/", one_hot=True)
  return mnist

class FullyConnectedLayer(object):
  def __init__(self, n_in, n_out, activation_fn, input, dtype=tf.float64):
    W = weight_variable([n_in, n_out])
    b = bias_variable([n_out])
    self.output = activation_fn(tf.matmul(input, W) + b)

def zeroLayerSoftmax(mnist, learning_rate=0.5, mini_batch_size=100, epochs=1000):
  ## setup variables
  x = tf.placeholder(tf.float64, [None, 784])
  h = FullyConnectedLayer(784, 100, tf.nn.sigmoid, x).output
  y = FullyConnectedLayer(100, 10, tf.nn.softmax, h).output
  y_ = tf.placeholder(tf.float64, [None, 10])
  ## determine cost fn
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  ## set learning rate
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
  sess = tf.InteractiveSession()
  ## init variables
  tf.global_variables_initializer().run()
  ## train for real
  for _ in range(epochs):
    batch_xs, batch_ys = mnist.train.next_batch(mini_batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  ## calculate accuracy
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def run(times, epochs):
  ret = []
  mnist = loadData()
  for _ in range(times):
    ret.append(zeroLayerSoftmax(mnist, epochs=epochs))
  print(sum(ret) / times)

if __name__ == "__main__":
  run(1, 1000)