import plotly.offline as py
import plotly.graph_objs as go
import numpy as np


def createtrace(x, y):
  """ It works with both array and ndarray """
  trace = go.Scatter(
    x=x,
    y=y
  )
  return trace


def drawonegraph(x, y, filename):
  data = [createtrace(x, y)]
  py.plot(data, filename=filename)


def drawmultigraph(x, ys, filename):
  data = []
  for y in ys:
    data.append(createtrace(x, y))
  py.plot(data, filename=filename)


def drawexperimentgraph(experiment, filename):
  x = range(1, 1 + experiment.epochs)
  y = experiment.output
  drawmultigraph(x, y, filename)


if __name__ == "__main__":
  x = [1, 2, 3, 4, 5]
  y1 = [1, 4, 9, 16, 25]
  drawonegraph(x, y1, "test1.html")
  y2 = [1, 2, 3, 4, 5]
  y = [y1, y2]
  drawmultigraph(x, y, "test2.html")
