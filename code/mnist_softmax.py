import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def loadData():
  ## load data
  mnist = input_data.read_data_sets("../input/MNIST_data/", one_hot=True)
  return mnist

def zeroLayerSoftmax(mnist,learning_rate=0.5, mini_batch_size=100, epochs=1000):
  ## setup variables
  x = tf.placeholder(tf.float32, [None, 784])
  W1 = weight_variable([784, 100])
  b1 = bias_variable([100])
  z1 = tf.matmul(x, W1) + b1
  a1 = tf.nn.sigmoid(z1)
  W = weight_variable([100, 10])
  b = bias_variable([10])
  y = tf.nn.softmax(tf.matmul(a1, W) + b)
  y_ = tf.placeholder(tf.float32, [None, 10])
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

ret = []
mnist = loadData()
print(zeroLayerSoftmax(mnist))
for _ in range(5):
  ret.append(zeroLayerSoftmax(mnist))
print(sum(ret)/5)

