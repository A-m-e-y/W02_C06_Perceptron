import numpy as np
import sys
from itertools import product
import matplotlib.pyplot as plt
import networkx as nx

# Sigmoid and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class MLP_XOR:
    def __init__(self, input_size, hidden_size=None, output_size=1, learning_rate=0.1):
        self.lr = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(input_size, self.hidden_size)
        self.bias_hidden = np.random.randn(self.hidden_size)

        self.weights_hidden_output = np.random.randn(self.hidden_size, output_size)
        self.bias_output = np.random.randn(output_size)

    def forward(self, x):
        self.input = x
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.final_input)

        return self.output

    def backward(self, target):
        output_error = target - self.output
        output_delta = output_error * sigmoid_derivative(self.final_input)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_input)

        self.weights_hidden_output += self.lr * np.outer(self.hidden_output, output_delta)
        self.bias_output += self.lr * output_delta

        self.weights_input_hidden += self.lr * np.outer(self.input, hidden_delta)
        self.bias_hidden += self.lr * hidden_delta

        return np.mean(output_error ** 2)

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            loss = 0
            for xi, target in zip(X, y):
                self.forward(xi)
                loss += self.backward(np.array([target]))
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, x):
        return self.forward(x)

    def draw_structure(self):
        G = nx.DiGraph()

        # Nodes
        input_nodes = [f"I{i+1}" for i in range(self.input_size)]
        hidden_nodes = [f"H{i+1}" for i in range(self.hidden_size)]
        output_nodes = [f"O{i+1}" for i in range(self.output_size)]

        G.add_nodes_from(input_nodes)
        G.add_nodes_from(hidden_nodes)
        G.add_nodes_from(output_nodes)

        # Edges
        for i in input_nodes:
            for h in hidden_nodes:
                G.add_edge(i, h)
        for h in hidden_nodes:
            for o in output_nodes:
                G.add_edge(h, o)

        # Positioning by layers
        pos = {}
        layer_spacing = 2
        node_spacing = 1

        # Center-align nodes vertically
        input_offset = -(len(input_nodes) - 1) * node_spacing / 2
        hidden_offset = -(len(hidden_nodes) - 1) * node_spacing / 2
        output_offset = -(len(output_nodes) - 1) * node_spacing / 2

        for i, node in enumerate(input_nodes):
            pos[node] = (0, input_offset + i * node_spacing)
        for i, node in enumerate(hidden_nodes):
            pos[node] = (layer_spacing, hidden_offset + i * node_spacing)
        for i, node in enumerate(output_nodes):
            pos[node] = (2 * layer_spacing, output_offset + i * node_spacing)

        # Draw graph
        plt.figure(figsize=(8, 6))
        nx.draw_networkx(G, pos, with_labels=True, arrows=True,
                         node_color='skyblue', node_size=1500,
                         edge_color='gray', font_size=12)

        plt.title(f"MLP Structure: {self.input_size} Input(s), {self.hidden_size} Hidden, {self.output_size} Output")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def generate_xor_data(n_inputs):
    X = np.array(list(product([0, 1], repeat=n_inputs)))
    y = np.array([x.sum() % 2 for x in X])  # XOR = parity of inputs
    return X, y

def main():
    if len(sys.argv) < 2:
        print("Usage: python xor_mlp_configurable.py <num_inputs> [-v]")
        sys.exit(1)

    n_inputs = int(sys.argv[1])
    X, y = generate_xor_data(n_inputs)

    print(f"Training XOR MLP with {n_inputs} inputs...")
    mlp = MLP_XOR(input_size=n_inputs)
    # mlp = MLP_XOR(input_size=n_inputs, hidden_size=4)
    mlp.train(X, y)

    print("\nPredictions:")
    for xi in X:
        output = mlp.predict(xi)
        print(f"Input: {xi}, Predicted: {output[0]:.4f}, Binary: {round(output[0])}")
    
    if "-v" in sys.argv:
        mlp.draw_structure()


if __name__ == "__main__":
    main()
