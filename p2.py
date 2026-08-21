import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
"""weights = [0.2, 0.8, -0.5, 1.0]
bias = 2"""

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]
           ]

biases = [2, 3, 0.5]




output = np.dot(weights, inputs) + biases #dot(weights[0], inputs) + biases[0], dot(weights[1], inputs) + biases[1], dot(weights[2], inputs) + biases[2]
print(output)