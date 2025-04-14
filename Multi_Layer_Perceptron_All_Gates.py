import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sys

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class LogicGateMLP:
    def __init__(self, input_size=2, hidden_size=4, output_size=6, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Weights and biases
        self.w_input_hidden = np.random.randn(input_size, hidden_size)
        self.b_hidden = np.random.randn(hidden_size)

        self.w_hidden_output = np.random.randn(hidden_size, output_size)
        self.b_output = np.random.randn(output_size)

    def forward(self, x):
        self.x = x
        self.hidden_input = np.dot(x, self.w_input_hidden) + self.b_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        self.output_input = np.dot(self.hidden_output, self.w_hidden_output) + self.b_output
        self.output = sigmoid(self.output_input)
        return self.output

    def backward(self, target):
        error_output = target - self.output
        delta_output = error_output * sigmoid_derivative(self.output_input)

        error_hidden = delta_output.dot(self.w_hidden_output.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden_input)

        # Update weights
        self.w_hidden_output += self.lr * np.outer(self.hidden_output, delta_output)
        self.b_output += self.lr * delta_output

        self.w_input_hidden += self.lr * np.outer(self.x, delta_hidden)
        self.b_hidden += self.lr * delta_hidden

        return np.sum(error_output**2)

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            total_loss = 0
            for xi, target in zip(X, y):
                self.forward(xi)
                total_loss += self.backward(target)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

    def predict(self, x):
        return self.forward(x)

    def draw_structure(self, output_labels=None):
        G = nx.DiGraph()

        input_nodes = [f"I{i+1}" for i in range(self.input_size)]
        hidden_nodes = [f"H{i+1}" for i in range(self.hidden_size)]
        if output_labels is None:
            output_nodes = [f"O{i+1}" for i in range(self.output_size)]
        else:
            output_nodes = output_labels

        G.add_nodes_from(input_nodes + hidden_nodes + output_nodes)

        # Input to Hidden
        for i in input_nodes:
            for h in hidden_nodes:
                G.add_edge(i, h)

        # Hidden to Output
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

        plt.figure(figsize=(10, 6))
        nx.draw_networkx(G, pos, with_labels=True, arrows=True,
                         node_color='lightblue', node_size=1500,
                         edge_color='gray', font_size=12)
        plt.title("Logic Gate MLP Structure")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    # Inputs: all 2-bit combinations
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Expected outputs: [NAND, AND, NOR, OR, XOR, XNOR]
    y = np.array([
        [1, 0, 1, 0, 0, 1],  # 0, 0
        [1, 0, 0, 1, 1, 0],  # 0, 1
        [1, 0, 0, 1, 1, 0],  # 1, 0
        [0, 1, 0, 1, 0, 1]   # 1, 1
    ])

    mlp = LogicGateMLP(input_size=2, hidden_size=4, output_size=6)

    if "-v" in sys.argv:
        mlp.draw_structure(["NAND", "AND", "NOR", "OR", "XOR", "XNOR"])

    print("Training logic gate MLP...")
    mlp.train(X, y)

    print("\nPredictions:")
    for xi in X:
        out = mlp.predict(xi)
        rounded = [round(o) for o in out]
        print(f"Input: {xi} → NAND:{rounded[0]}, AND:{rounded[1]}, NOR:{rounded[2]}, OR:{rounded[3]}, XOR:{rounded[4]}, XNOR:{rounded[5]}")

if __name__ == "__main__":
    main()
