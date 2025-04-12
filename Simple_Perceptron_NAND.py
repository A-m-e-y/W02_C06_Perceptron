import numpy as np

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
        self.weights = np.zeros(input_size)
        self.bias = 0.0
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

def main():
    # Step 3: NAND data
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([1, 1, 1, 0])  # NAND outputs

    # Create perceptron
    p = Perceptron(input_size=2, learning_rate=0.1)

    # Train perceptron
    print("Training the perceptron for NAND logic:")
    p.train(X, y)

    # Test predictions
    print("\nTesting predictions:")
    for xi in X:
        output = p.predict(xi)
        print(f"Input: {xi}, Predicted: {output:.4f}, Binary: {round(output)}")

if __name__ == "__main__":
    main()
