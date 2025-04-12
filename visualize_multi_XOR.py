import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Multi_Layer_Perceptron_XOR import MLP_XOR
from matplotlib.animation import PillowWriter  # Ensure PillowWriter is imported

# === Generate XOR data ===
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])

# === Train the model and store snapshots ===
mlp = MLP_XOR(input_size=2, hidden_size=2, learning_rate=0.1)
epochs = 10000
frames = 30
epochs_per_frame = epochs // frames
model_snapshots = []

# Train and capture snapshots
for epoch in range(epochs):
    for xi, target in zip(X, y):
        mlp.forward(xi)
        mlp.backward(np.array([target]))
    
    if epoch % epochs_per_frame == 0:
        snapshot = (
            mlp.weights_input_hidden.copy(),
            mlp.bias_hidden.copy(),
            mlp.weights_hidden_output.copy(),
            mlp.bias_output.copy()
        )
        model_snapshots.append(snapshot)

# === Setup plot ===
fig, ax = plt.subplots()
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

def update(frame_idx):
    # Clear the entire plot
    ax.clear()

    # Restore snapshot
    wi, bi, wo, bo = model_snapshots[frame_idx]
    mlp.weights_input_hidden = wi
    mlp.bias_hidden = bi
    mlp.weights_hidden_output = wo
    mlp.bias_output = bo

    # Predict over grid
    zz = np.array([mlp.predict(p)[0] for p in grid_points])
    zz = zz.reshape(xx.shape)

    # Plot contour
    ax.contourf(xx, yy, zz, levels=100, cmap='coolwarm', alpha=0.8)

    # Plot XOR data
    for xi, label in zip(X, y):
        color = 'black' if label == 1 else 'white'
        ax.scatter(*xi, c=color, edgecolor='k', s=150)

    # Labels and title
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_title(f"Epoch {(frame_idx + 1) * epochs_per_frame}")

    # Fix legend icons
    class_0 = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=10, label="0 : WHITE")
    class_1 = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label="1 : BLACK")
    ax.legend(handles=[class_0, class_1], loc="upper left")


# Run the animation (no blit)
ani = FuncAnimation(fig, update, frames=len(model_snapshots), interval=0.1, repeat=False)

# Save the animation as a GIF
gif_filename = "animations/multi_XOR_perceptron_animation.gif"
print(f"Saving animation as {gif_filename}...")
ani.save(gif_filename, writer=PillowWriter(fps=60))  # Save with 15 frames per second
print(f"Animation saved as {gif_filename}.")

plt.tight_layout()
plt.show()
