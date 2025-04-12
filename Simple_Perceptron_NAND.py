import numpy as np
import sys

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Perceptron class using sigmoid activation
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
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
                total_loss += error ** 2

                # Perceptron weight update rule
                d_output = error * sigmoid_derivative(z)
                self.weights += self.lr * d_output * xi
                self.bias += self.lr * d_output

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

def generate_nand_data(n_inputs):
    """Generate training data for an N-input NAND gate."""
    X = np.array([list(map(int, f"{i:0{n_inputs}b}")) for i in range(2**n_inputs)])
    y = np.array([1 if np.sum(x) < n_inputs else 0 for x in X])  # NAND logic
    return X, y

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 Simple_Perceptron_NAND.py <number_of_inputs>")
        sys.exit(1)

    try:
        n_inputs = int(sys.argv[1])
        if n_inputs < 2:
            raise ValueError("Number of inputs must be at least 2.")
    except ValueError as e:
        print(f"Invalid input: {e}")
        sys.exit(1)

    # Generate training data for N-input NAND gate
    X, y = generate_nand_data(n_inputs)

    # Create perceptron
    p = Perceptron(input_size=n_inputs, learning_rate=0.1)

    # Train perceptron
    print(f"Training the perceptron for {n_inputs}-input NAND logic:")
    p.train(X, y)

    # Test predictions
    print("\nTesting predictions:")
    for xi in X:
        output = p.predict(xi)
        print(f"Input: {xi}, Predicted: {output:.4f}, Binary: {round(output)}")

if __name__ == "__main__":
    main()
