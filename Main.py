import numpy

import Neural_network as Ai

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3

NN = Ai.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print(NN.query([1.0, 0.5, -1.5]))
print("Every thing is ok!")