import plotly.offline as py
import plotly.graph_objs as go
import numpy as np


def createtrace(x, y, name=""):
  """ It works with both array and ndarray """
  if name == "":
    trace = go.Scatter(
      x=x,
      y=y
    )
  else:
    trace = go.Scatter(
      x=x,
      y=y,
      name=name
    )
  return trace


def drawonegraph(x, y, filename):
  data = [createtrace(x, y)]
  py.plot(data, filename=filename + ".html")


def drawmultigraph(x, ys, filename, labels, title, xaxis, yaxis):
  data = []
  for y, label in zip(ys, labels):
    data.append(createtrace(x, y, label))
  layout = go.Layout(
    title=title,
    xaxis=dict(title=xaxis),
    yaxis=dict(title=yaxis)
  )
  fig = go.Figure(data=data, layout=layout)
  py.plot(fig, filename=filename + ".html")


def drawexperimentgraph(experiment, filename):
  y = experiment.output
  n_point = len(y[0])
  x = list(range(1, 2 + n_point))
  drawmultigraph(x, y, filename, experiment.tags, experiment.name, "epochs", "accuracy")


if __name__ == "__main__":
  x = [1, 2, 3, 4, 5]
  y1 = [1, 4, 9, 16, 25]
  drawonegraph(x, y1, "test1")
  y2 = [1, 2, 3, 4, 5]
  y = [y1, y2]
  drawmultigraph(x, y, "test2")
