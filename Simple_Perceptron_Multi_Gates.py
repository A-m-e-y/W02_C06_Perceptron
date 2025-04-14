import numpy as np
import sys
import matplotlib.pyplot as plt
import networkx as nx

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Multi-output perceptron (no hidden layer)
class MultiOutputPerceptron:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)
        self.lr = learning_rate

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return sigmoid(z)

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            total_loss = 0
            for xi, target in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                output = sigmoid(z)
                error = target - output
                total_loss += np.sum(error**2)

                delta = error * sigmoid_derivative(z)
                self.weights += self.lr * np.outer(xi, delta)
                self.bias += self.lr * delta

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")
    
    def draw_structure(self, output_labels=None):
        G = nx.DiGraph()

        if output_labels is None:
            output_labels = [f"O{i+1}" for i in range(self.output_size)]

        input_nodes = [f"I{i+1}" for i in range(self.input_size)]
        output_nodes = output_labels

        G.add_nodes_from(input_nodes)
        G.add_nodes_from(output_nodes)

        for i in input_nodes:
            for o in output_nodes:
                G.add_edge(i, o)

        # Positioning by layers
        pos = {}
        layer_spacing = 2
        node_spacing = 1

        # Center-align nodes vertically
        input_offset = -(len(input_nodes) - 1) * node_spacing / 2
        output_offset = -(len(output_nodes) - 1) * node_spacing / 2

        for i, node in enumerate(input_nodes):
            pos[node] = (0, input_offset + i * node_spacing)
        for i, node in enumerate(output_nodes):
            pos[node] = (layer_spacing, output_offset + i * node_spacing)

        plt.figure(figsize=(8, 5))
        nx.draw_networkx(G, pos, with_labels=True, arrows=True,
                         node_color='lightblue', node_size=1500,
                         edge_color='gray', font_size=12)
        plt.title(f"Perceptron Structure: {self.input_size} Inputs → {self.output_size} Outputs")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# === Main ===
def main():
    # Input combinations (2-input binary)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Expected outputs: [NAND, AND, NOR, OR]
    y = np.array([
        [1, 0, 1, 0],  # 0, 0
        [1, 0, 0, 1],  # 0, 1
        [1, 0, 0, 1],  # 1, 0
        [0, 1, 0, 1]   # 1, 1
    ])

    p = MultiOutputPerceptron(input_size=2, output_size=4, learning_rate=0.1)
    print("Training multi-output perceptron for NAND, AND, NOR, OR:")
    p.train(X, y)

    if "-v" in sys.argv:
        p.draw_structure(output_labels=["NAND", "AND", "NOR", "OR"])
    # Final predictions
    print("\nPredictions:")
    for xi in X:
        out = p.predict(xi)
        print(f"Input: {xi} → NAND: {round(out[0])}, AND: {round(out[1])}, NOR: {round(out[2])}, OR: {round(out[3])}")

if __name__ == "__main__":
    main()
