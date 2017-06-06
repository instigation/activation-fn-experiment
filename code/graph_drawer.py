import plotly.offline as py
import plotly.graph_objs as go
import numpy as np


def createTrace(x, y):
  """ It works with both array and ndarray """
  trace = go.Scatter(
    x=x,
    y=y
  )
  return trace


def drawOneGraph(x, y, filename):
  data = [createTrace(x, y)]
  py.plot(data, filename=filename)


def drawMultiGraph(x, ys, filename):
  data = []
  for y in ys:
    data.append(createTrace(x, y))
  py.plot(data, filename=filename)


def drawExperimentGraph(experiment, filename):
  x = range(1, 1 + experiment.epochs)
  y = experiment.output
  drawMultiGraph(x, y, filename)


if __name__ == "__main__":
  x = [1, 2, 3, 4, 5]
  y1 = [1, 4, 9, 16, 25]
  drawOneGraph(x, y1, "test1.html")
  y2 = [1, 2, 3, 4, 5]
  y = [y1, y2]
  drawMultiGraph(x, y, "test2.html")
