import tensorflow as tf


class FullyConnectedLayer(object):
  def __init__(self, n_in, n_out, activation_fn, var_in, dtype):
    self.n_in = n_in
    self.n_out = n_out
    self.activation_fn = activation_fn
    self.var_in = var_in
    W = self.smartrand_weight_variable([n_in, n_out], n_out, dtype)
    b = self.zero_bias_variable([n_out], dtype)
    self.var_out = activation_fn(tf.matmul(var_in, W) + b)

  def smartrand_weight_variable(self, shape, n_out, dtype):
    initial = tf.truncated_normal(shape, stddev=1 / n_out, dtype=dtype)
    return tf.Variable(initial, dtype=dtype)

  def zero_bias_variable(self, shape, dtype):
    initial = tf.constant(0.0, shape=shape, dtype=dtype)
    return tf.Variable(initial, dtype=dtype)


class Network(object):
  def __init__(self, layers, cost_fn):
    self.layers = layers
    self.cost_fn = cost_fn
    self.y_ = tf.placeholder(tf.float64, [None, layers[-1].n_out])
    self.y = layers[-1].var_out
    self.x = layers[0].var_in
    self.cost = self.cost_fn(self.y, self.y_)
    self.learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])
    self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate_placeholder).minimize(self.cost)
    correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def calculateaccuracy(self, sess, x, y):
    return sess.run(self.accuracy, feed_dict={self.x: x, self.y_: y})

  def runmnist(self, mini_batch_size, epochs, learning_rate, mnist, verboes=True, savelog=False, logdir='logs',
               printfreq=1000):
    with tf.Session() as sess:
      if savelog:
        writer = tf.summary.FileWriter(logdir, sess.graph)
        writer.close()
      tf.global_variables_initializer().run()
      accuracies = []
      for epoch_index in range(epochs):
        batch_xs, batch_ys = mnist.train.next_batch(mini_batch_size)
        sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys, self.learning_rate_placeholder: learning_rate})
        if (epoch_index + 1) % printfreq == 0:
          accuracy = self.calculateaccuracy(sess, mnist.test.images, mnist.test.labels)
          accuracies.append(accuracy)
          if verboes:
            print("epochs %d : %.2f" % (epoch_index + 1, accuracy * 100) + '%')

      return accuracies


if __name__ == "__main__":
  from code.data import Data


  def reluVar(a):
    return lambda x: tf.maximum(x * a, tf.constant(0.0, dtype=tf.float64))


  def transSigmoid(down):
    return lambda x: tf.nn.sigmoid(x) - tf.constant(down, dtype=tf.float64)


  dtype = tf.float64
  x = tf.placeholder(dtype, [None, 784])
  layer01 = FullyConnectedLayer(784, 100, transSigmoid(0.5), x, dtype)
  layer12 = FullyConnectedLayer(100, 10, tf.nn.softmax, layer01.var_out, dtype)


  def cross_entropy(y, y_):
    return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


  net = Network([layer01, layer12], cross_entropy)
  mnist = Data().loadmnistdata()
  net.runmnist(100, 20000, 0.5, mnist)
