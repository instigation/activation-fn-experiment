import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle

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

def myrelu(x):
  return tf.maximum(x, tf.constant(0.0, dtype=tf.float64))

def zeroLayerSoftmax(mnist, learning_rate=0.5, mini_batch_size=100, epochs=1000):
  ## setup variables
  x = tf.placeholder(tf.float64, [None, 784])
  h_fc1 = FullyConnectedLayer(784, 10, tf.nn.sigmoid, x).output
  y = FullyConnectedLayer(10, 10, tf.nn.softmax, h_fc1).output
  y_ = tf.placeholder(tf.float64, [None, 10])
  ## determine cost fn
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  ## set learning rate
  learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])
  train_step = tf.train.GradientDescentOptimizer(learning_rate_placeholder).minimize(cross_entropy)
  sess = tf.InteractiveSession()
  ## init variables
  tf.global_variables_initializer().run()

  def calculateAccuracy():
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return sess.run(accuracy, feed_dict={x:mnist.test.images, y_: mnist.test.labels})

  ## train for real
  previous_accuracy = 0.0
  standard = 0.0
  accumulative_accuracy = []
  for epoch_index in range(epochs):
    batch_xs, batch_ys = mnist.train.next_batch(mini_batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate_placeholder: learning_rate})
    if epoch_index % 1000 == 999:
      current_accuracy = calculateAccuracy()
      accumulative_accuracy.append(current_accuracy)
      print(current_accuracy)
      if abs(previous_accuracy - current_accuracy) < standard:
        print("adapting learning rate...")
        learning_rate /= 3
        standard /= 5
      previous_accuracy = current_accuracy

  return accumulative_accuracy

def run(times, epochs):
  ret = []
  mnist = loadData()
  for _ in range(times):
    acc = zeroLayerSoftmax(mnist, epochs=epochs)
    ret.append(acc[len(acc) - 1])
  print(sum(ret) / times)
  return ret

def runSaveData(times, epochs):
  ret = []
  mnist = loadData()
  for _ in range(times):
    acc = zeroLayerSoftmax(mnist, epochs=epochs)
    ret.append(acc)
  np.savetxt("../output/test.txt", ret)

def loadArray(filepath="../output/test.txt"):
  return np.loadtxt(filepath)

def runGraph(epochs):
  mnist = loadData()
  return zeroLayerSoftmax(mnist, epochs=epochs)

if __name__ == "__main__":
  mini_batch_size = 1000
  epochs = 10
  times = 5
  runSaveData(times, mini_batch_size * epochs)
  ret = loadArray()
  print(ret)
  from code.graph_drawer import drawGraphByEpochs
  drawGraphByEpochs(ret, epochs)
