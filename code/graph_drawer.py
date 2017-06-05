import plotly.offline as py
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

def drawGraph(x, y):
    # Create a trace
    trace = go.Scatter(
        x=x,
        y=y
    )

    data = [trace]

    py.plot(data, filename='basic-line')

if __name__ == "__main__":
    from code.mnist_softmax import runGraph
    accuracy = np.asarray(runGraph(30000))
    epochs = np.asarray(range(30))
    drawGraph(epochs, accuracy)