from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os


class Data(object):
  def __init__(self, input_dir="../input", output_dir="../output"):
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.mnist_dir = os.path.join(input_dir, "MNIST_data")

  def loadmnistdata(self):
    ## load data
    mnist = input_data.read_data_sets(self.mnist_dir, one_hot=True)
    return mnist

  def savetxt(self, filename, object):
    np.savetxt(os.path.join(self.output_dir, filename), object)

  def loadoutputtxt(self, filename):
    return np.loadtxt(os.path.join(self.output_dir, filename))
