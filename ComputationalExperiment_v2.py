"""Reproducible computations for the ANN mathematical exposition.

Network: 2 inputs -> 2 sigmoid hidden units -> 1 sigmoid output; no biases.
Weight ordering: (w1,w2,w3,w4,w5,w6), with hidden connections
h=x1*w1+x2*w3, k=x1*w2+x2*w4 and output m=a*w5+b*w6.
Loss: 0.5*(target-output)**2 for one input-target pair.

Run: python ComputationalExperiment_Publication.py
Requires: numpy, matplotlib
Output: PNG/PDF figures, CSV tables, and metadata in ann_results/.
"""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a graphical display.
import matplotlib.pyplot as plt
import numpy as np

X1, X2, TARGET = 0.35, 0.9, 0.8
INITIAL_WEIGHTS = np.array([0.1, 0.4, 0.8, 0.6, 0.3, 0.9], dtype=np.float64)
LEARNING_RATES = (0.01, 0.1, 0.3, 0.8)
ITERATIONS = 1000
EPSILON = 1e-5
OUTPUT_DIR = Path(__file__).resolve().parent / "ann_results"


def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def forward(weights: np.ndarray) -> tuple[float, float, float, float, float, float]:
    w1, w2, w3, w4, w5, w6 = weights
    h = X1 * w1 + X2 * w3
    k = X1 * w2 + X2 * w4
    a, b = sigmoid(h), sigmoid(k)
    m = a * w5 + b * w6
    return float(h), float(k), a, b, float(m), sigmoid(m)


def loss_from_output(output: float) -> float:
    return 0.5 * (TARGET - output) ** 2


def loss_from_weights(weights: np.ndarray) -> float:
    return loss_from_output(forward(weights)[-1])


def analytical_gradients(weights: np.ndarray) -> np.ndarray:
    _, _, a, b, _, y = forward(weights)
    delta_output = (y - TARGET) * y * (1.0 - y)
    delta_h = delta_output * weights[4] * a * (1.0 - a)
    delta_k = delta_output * weights[5] * b * (1.0 - b)
    return np.array([
        delta_h * X1, delta_k * X1,
        delta_h * X2, delta_k * X2,
        delta_output * a, delta_output * b,
    ], dtype=np.float64)


def numerical_gradients(weights: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    gradients = np.empty_like(weights)
    for i in range(len(weights)):
        plus, minus = weights.copy(), weights.copy()
        plus[i] += epsilon
        minus[i] -= epsilon
        gradients[i] = (loss_from_weights(plus) - loss_from_weights(minus)) / (2.0 * epsilon)
    return gradients


def train(rate: float, iterations: int = ITERATIONS) -> dict:
    weights = INITIAL_WEIGHTS.copy()
    losses = [loss_from_weights(weights)]  # Iteration 0: before any update.
    for _ in range(iterations):
        weights -= rate * analytical_gradients(weights)  # All six updated simultaneously.
        losses.append(loss_from_weights(weights))
    output = forward(weights)[-1]
    return {
        "rate": rate, "weights": weights, "output": output,
        "final_loss": losses[-1], "losses": np.asarray(losses),
        "absolute_error": abs(TARGET - output),
        "loss_reduction_percent": (1.0 - losses[-1] / losses[0]) * 100.0,
    }


def write_csv(path: Path, headings: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headings)
        writer.writerows(rows)


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    initial_y = forward(INITIAL_WEIGHTS)[-1]
    initial_loss = loss_from_weights(INITIAL_WEIGHTS)
    one_step_weights = INITIAL_WEIGHTS - 0.3 * analytical_gradients(INITIAL_WEIGHTS)
    one_step_y = forward(one_step_weights)[-1]
    one_step_loss = loss_from_weights(one_step_weights)

    print("INITIAL NETWORK AND ONE STEP (learning rate = 0.3)")
    print(f"Initial output: {initial_y:.12f}; initial loss: {initial_loss:.12e}")
    print(f"After one update: {one_step_y:.12f}; loss: {one_step_loss:.12e}\n")
    write_csv(OUTPUT_DIR / "single_step.csv",
              ["state", "network_output", "half_squared_error_loss"],
              [["before", repr(initial_y), repr(initial_loss)],
               ["after", repr(one_step_y), repr(one_step_loss)]])

    analytic = analytical_gradients(INITIAL_WEIGHTS)
    numeric = numerical_gradients(INITIAL_WEIGHTS)
    difference = np.abs(analytic - numeric)
    print("GRADIENT CHECK (central difference, epsilon = 1e-5)")
    print("weight       analytical             numerical             absolute difference")
    for i in range(6):
        print(f"w{i+1}    {analytic[i]: .12e}    {numeric[i]: .12e}    {difference[i]:.6e}")
    print(f"Maximum absolute difference: {np.max(difference):.6e}\n")
    assert np.max(difference) < 1e-8, "Gradient check failed; inspect derivatives."
    write_csv(OUTPUT_DIR / "gradient_check.csv",
              ["weight", "analytical_gradient", "numerical_gradient", "absolute_difference"],
              [[f"w{i+1}", repr(analytic[i]), repr(numeric[i]), repr(difference[i])]
               for i in range(6)])

    runs = [train(rate) for rate in LEARNING_RATES]
    print("LEARNING RATE COMPARISON (1000 updates)")
    for run in runs:
        print(f"eta={run['rate']:<4}  output={run['output']:.12f}  "
              f"loss={run['final_loss']:.12e}  abs_error={run['absolute_error']:.12e}")
    write_csv(OUTPUT_DIR / "learning_rate_summary.csv",
              ["learning_rate", "updates", "final_output", "final_loss", "absolute_error", "loss_reduction_percent"],
              [[r["rate"], ITERATIONS, repr(r["output"]), repr(r["final_loss"]),
                repr(r["absolute_error"]), repr(r["loss_reduction_percent"])] for r in runs])
    write_csv(OUTPUT_DIR / "loss_trajectories.csv",
              ["iteration"] + [f"loss_eta_{r['rate']}" for r in runs],
              [[i] + [repr(r["losses"][i]) for r in runs] for i in range(ITERATIONS + 1)])

    # Figure 12: Before/after single gradient descent step (paper numbering).
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    bars = ax.bar(["Before update", "After update"], [initial_loss, one_step_loss], width=0.55)
    ax.set_ylabel("Half squared-error loss")
    ax.set_title("Loss before and after one gradient-descent update")
    ax.set_ylim(0, initial_loss * 1.18)
    for bar, value in zip(bars, [initial_loss, one_step_loss]):
        ax.annotate(f"{value:.8f}", (bar.get_x() + bar.get_width()/2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "Figure_12_single_step_loss")

    # Figure 13: 1000 updates plus iteration zero.
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for run in runs:
        ax.plot(np.arange(ITERATIONS + 1), run["losses"], label=f"η = {run['rate']}", linewidth=1.8)
    ax.set_xlabel("Gradient-descent updates")
    ax.set_ylabel("Half squared-error loss")
    ax.set_title("Effect of learning rate on training loss")
    ax.set_xlim(0, ITERATIONS)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_figure(fig, "Figure_13_learning_rate_loss")

    # Figure 14: log axis so tiny but nonzero errors remain visible.
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    x = np.arange(len(runs))
    errors = [r["absolute_error"] for r in runs]
    bars = ax.bar(x, errors, width=0.55)
    ax.set_xticks(x, [str(r["rate"]) for r in runs])
    ax.set_yscale("log")
    ax.set_xlabel("Learning rate (η)")
    ax.set_ylabel("Absolute prediction error |target − output| (log scale)")
    ax.set_title("Final prediction error after 1,000 updates")
    ax.set_ylim(min(errors)/10, max(errors)*20)
    for bar, value in zip(bars, errors):
        ax.annotate(f"{value:.2e}", (bar.get_x()+bar.get_width()/2, value),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.grid(axis="y", which="both", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "Figure_14_absolute_prediction_error")

    metadata = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "network": "2-2-1 sigmoid, no biases",
        "weight_order": ["w1", "w2", "w3", "w4", "w5", "w6"],
        "initial_weights": INITIAL_WEIGHTS.tolist(),
        "inputs": [X1, X2], "target": TARGET,
        "loss": "0.5*(target-output)**2",
        "learning_rates": list(LEARNING_RATES),
        "updates_per_run": ITERATIONS,
        "gradient_check_epsilon": EPSILON,
        "arithmetic": "numpy.float64 / Python double-precision floating point",
        "random_seed": None,
        "randomness": "none: all inputs and initial weights are fixed",
        "trajectory_includes_iteration_zero": True,
    }
    (OUTPUT_DIR / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved figures, CSV files and metadata to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()