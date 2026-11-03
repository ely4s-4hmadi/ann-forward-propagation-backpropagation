import numpy as np
import matplotlib.pyplot as plt

# Input values and target
x1, x2 = 0.35, 0.9
target = 0.8

# Initial weights: w1, w2, w3, w4, w5, w6
initial_weights = np.array([
    0.1, 0.4, 0.8, 0.6, 0.3, 0.9
], dtype=float)

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward propagation
def forward(w):
    w1, w2, w3, w4, w5, w6 = w

    h = x1*w1 + x2*w3
    k = x1*w2 + x2*w4

    a = sigmoid(h)
    b = sigmoid(k)

    m = a*w5 + b*w6
    y = sigmoid(m)

    return h, k, a, b, m, y

# Loss function
def calculate_loss(y):
    return 0.5 * (target - y)**2

# Run one training experiment
def train(learning_rate, iterations=1000):
    w = initial_weights.copy()
    losses = []

    for _ in range(iterations):
        h, k, a, b, m, y = forward(w)

        loss = calculate_loss(y)
        losses.append(loss)

        # Output-layer gradient
        delta_output = (y - target) * y * (1 - y)

        # Hidden-layer gradients
        delta_h = delta_output * w[4] * a * (1 - a)
        delta_k = delta_output * w[5] * b * (1 - b)

        # Gradients of all six weights
        gradients = np.array([
            delta_h * x1,       # dw1
            delta_k * x1,       # dw2
            delta_h * x2,       # dw3
            delta_k * x2,       # dw4
            delta_output * a,   # dw5
            delta_output * b    # dw6
        ])

        # Update all weights simultaneously
        w = w - learning_rate * gradients

    # Record the final loss
    _, _, _, _, _, final_y = forward(w)
    losses.append(calculate_loss(final_y))

    return w, final_y, losses


# Compare different learning rates
learning_rates = [0.01, 0.1, 0.3, 0.8]
results = {}

for rate in learning_rates:
    weights, output, losses = train(rate)
    results[rate] = losses

    print(
        f"Learning rate: {rate} | "
        f"Final output: {output:.8f} | "
        f"Final loss: {losses[-1]:.10f}"
    )

# Plot the loss curves
for rate, losses in results.items():
    plt.plot(losses, label=f"Learning rate = {rate}")

plt.xlabel("Training iterations")
plt.ylabel("Half squared-error loss")
plt.title("Effect of Learning Rate on Neural Network Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()