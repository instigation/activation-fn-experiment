import plotly.offline as py
import plotly.graph_objs as go

# Create random data with numpy
import numpy as np

def drawGraph(x, y):
    # Create a trace
    data = []
    for y0 in y:
        trace = go.Scatter(
            x=x,
            y=y0
        )
        data.append(trace)

    py.plot(data, filename='test.html')

def drawGraphByEpochs(y, epochs):
    epochs = np.asarray(range(1, epochs + 1))
    drawGraph(epochs, y)

if __name__ == "__main__":
    from code.mnist_softmax import runGraph
    mini_batch_size = 1000
    epochs = 10
    accuracy = np.asarray(runGraph(mini_batch_size * epochs))
    drawGraphByEpochs(accuracy, epochs)