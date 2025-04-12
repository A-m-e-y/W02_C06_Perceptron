import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from Simple_Perceptron_NOR import Perceptron

def main():
    # Fixed 2-input NOR gate
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([1, 0, 0, 0])  # NOR outputs

    # Initialize perceptron
    p = Perceptron(input_size=2, learning_rate=0.1)

    # Storage for animation frames (weights, bias after each epoch)
    history = []

    # Train and record weights after each epoch
    epochs = 500
    for epoch in range(epochs):
        for xi, target in zip(X, y):
            z = np.dot(xi, p.weights) + p.bias
            output = 1 / (1 + np.exp(-z))  # sigmoid
            error = target - output
            d_output = error * output * (1 - output)
            p.weights += p.lr * d_output * xi
            p.bias += p.lr * d_output

        # Store weights and bias for animation
        history.append((p.weights.copy(), p.bias))

    # === Plotting and Animation ===
    fig, ax = plt.subplots()
    colors = ['red' if label == 0 else 'blue' for label in y]
    sc = ax.scatter(X[:, 0], X[:, 1], c=colors, s=100, edgecolor='k')
    line, = ax.plot([], [], 'k--', lw=2)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_title("Learning NOR: Perceptron Decision Boundary")

    # Fix legend icons
    class_0 = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label="0 : RED")
    class_1 = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label="1 : BLUE")
    ax.legend(handles=[class_0, class_1], loc="upper left")


    def update(i):
        w, b = history[i]
        x_vals = np.array([-0.5, 1.5])
        if w[1] != 0:
            y_vals = -(w[0] * x_vals + b) / w[1]
        else:
            y_vals = np.array([0, 0])  # fallback
        line.set_data(x_vals, y_vals)
        ax.set_title(f"Epoch {i+1}/{epochs}")
        return line,

    ani = FuncAnimation(fig, update, frames=len(history), interval=0.1, repeat=False)  # Reduced interval to 1ms

    # Save animation as GIF
    gif_filename = "NOR_perceptron_animation.gif"
    print(f"Saving animation as {gif_filename}...")
    ani.save(gif_filename, writer=PillowWriter(fps=15))  # Save with 30 frames per second
    print(f"Animation saved as {gif_filename}.")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
