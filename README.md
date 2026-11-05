# A Mathematical Exposition of Forward Propagation and Backpropagation in Artificial Neural Networks

**Author:** Mohammad Elyas Ahmadi  
**Affiliation:** School of Computer Science and Informatics, University of Liverpool  
**Version:** 1.0 — November 2026

## Overview

This repository accompanies the independent mathematical exposition *A Mathematical Exposition of Forward Propagation and Backpropagation in Artificial Neural Networks*.

It provides Python implementations of a feedforward artificial neural network consisting of two input neurons, two hidden neurons, and one output neuron (2–2–1 architecture).

The implementations demonstrate the mathematical principles of forward propagation, backpropagation, gradient descent, and loss minimisation. The extended implementation additionally provides numerical gradient verification using central finite differences, a single-step weight-update demonstration, experimental visualisations, and exported numerical results.

The objective is to connect the mathematical derivations presented in the paper with reproducible computational evidence.

## Experimental Configuration

The computational experiments use the following configuration:

| Parameter | Value |
|---|---|
| Network architecture | 2–2–1 feedforward network |
| Activation function | Sigmoid |
| Loss function | Half squared error |
| Input values | 0.35, 0.9 |
| Target output | 0.8 |
| Initial weights | [0.1, 0.4, 0.8, 0.6, 0.3, 0.9] |
| Learning rates | 0.01, 0.1, 0.3, 0.8 |
| Training iterations | 1,000 |
| Single-step learning rate | 0.3 |
| Gradient-checking method | Central finite differences |
| Finite-difference step size | 0.00001 |

Biases are omitted from the numerical network to keep the mathematical derivations and calculations manageable. All experiments use the same initial weights and input-target pair.

## Python Implementations

Two Python implementations are provided to support both mathematical understanding and computational reproducibility.

### 1. Original Implementation — `ComputationalExperiment.py`

This is the original, concise implementation of the computational experiment.

It includes:

- Forward propagation through the 2–2–1 network.
- Sigmoid activation functions.
- Half squared-error loss calculation.
- Analytical gradient calculations using backpropagation.
- Simultaneous gradient-descent updates for all six weights.
- Comparison of four learning rates over 1,000 training iterations.
- Visualisation of the resulting loss trajectories.

This implementation is intended for readers who wish to understand how the mathematical expressions translate into a relatively simple Python program.

### 2. Extended Implementation — `ComputationalExperiment_v2.py`

This is the extended implementation intended for reproducing the computational results presented in the paper.

In addition to the original training experiment, it includes:

- Detailed forward-propagation calculations.
- A single gradient-descent update using a learning rate of 0.3.
- Analytical gradient calculations for all six weights.
- Numerical gradient verification using central finite differences.
- Comparison of analytical and numerical gradients.
- Learning-rate experiments over 1,000 iterations.
- Generation of Figures 12, 13, and 14.
- Export of numerical results to CSV files.
- Recording of experimental configuration and software information.

**The extended implementation is recommended for reproducing the numerical results and experimental figures presented in the paper.**

## Requirements

The implementations require:

- Python 3
- NumPy
- Matplotlib

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Reproducing the Experiments

Clone the repository:

```bash
git clone https://github.com/ely4s-4hmadi/ann-forward-propagation-backpropagation.git
```

Navigate to the repository:

```bash
cd ann-forward-propagation-backpropagation
```

### Running the Original Implementation

```bash
python ComputationalExperiment.py
```

This runs the four learning-rate experiments, prints the final network outputs and losses, and displays the loss trajectories.

### Running the Extended Implementation

```bash
python ComputationalExperiment_v2.py
```

This runs the extended numerical calculations, performs gradient verification, and generates the experimental figures and numerical result files.

By default, the extended script saves its generated outputs to an `ann_results/` directory.

For convenient access, the repository also contains organised copies of the experimental figures in `figures/` and the numerical results in `results/`.

## Repository Structure

```text
ann-forward-propagation-backpropagation/
│
├── ComputationalExperiment.py
├── ComputationalExperiment_v2.py
├── README.md
├── requirements.txt
│
├── figures/
│   ├── Figure_12_single_step_loss.png
│   ├── Figure_12_single_step_loss.pdf
│   ├── Figure_13_learning_rate_loss.png
│   ├── Figure_13_learning_rate_loss.pdf
│   ├── Figure_14_absolute_prediction_error.png
│   └── Figure_14_absolute_prediction_error.pdf
│
└── results/
    ├── gradient_check.csv
    ├── learning_rate_summary.csv
    ├── loss_trajectories.csv
    ├── single_step.csv
    └── experiment_metadata.json
```

The directory structure separates the source code, experimental visualisations, and numerical data to make the computational evidence easier to inspect and reproduce.

## Generated Figures

The extended implementation generates three figures corresponding to the computational experiment in Section 8 of the paper.

### Figure 12 — Loss Before and After One Gradient-Descent Update

Illustrates the reduction in half squared-error loss following one simultaneous update of the six weights using a learning rate of 0.3.

### Figure 13 — Effect of Learning Rate on Neural Network Training

Compares the loss trajectories for learning rates of 0.01, 0.1, 0.3, and 0.8 over 1,000 gradient-descent updates.

### Figure 14 — Absolute Prediction Error After Training

Compares the final absolute prediction errors for the four learning rates. A logarithmic vertical scale is used to display differences across several orders of magnitude.

Figures are provided in PNG and PDF formats for convenient viewing and inclusion in academic documents.

## Numerical Results

The `results/` directory contains the numerical data associated with the experiments.

| File | Description |
|---|---|
| `gradient_check.csv` | Analytical and numerical gradients, together with their absolute differences |
| `learning_rate_summary.csv` | Final predictions and losses for each learning rate |
| `loss_trajectories.csv` | Loss values recorded throughout training |
| `single_step.csv` | Results of the single gradient-descent update |
| `experiment_metadata.json` | Experimental configuration and software information |

These files allow readers to examine the underlying numerical values independently of the generated figures.

## Numerical Gradient Verification

To assess the correctness of the analytical backpropagation gradients, the extended implementation uses central finite differences.

For a weight $w_i$, the numerical gradient is approximated by:

$$
\frac{\partial L}{\partial w_i}
\approx
\frac{
L(\mathbf{w}+\varepsilon\mathbf{e}_i)
-
L(\mathbf{w}-\varepsilon\mathbf{e}_i)
}{
2\varepsilon
}
$$

where:

- $\varepsilon = 10^{-5}$ is the small perturbation used in the finite-difference approximation.
- $\mathbf{w}$ is the vector of network weights.
- $\mathbf{e}_i$ is the unit vector corresponding to weight $w_i$.
- $L$ represents the loss function.

The numerical gradients are compared with the **analytical gradients** obtained through backpropagation.

For the selected initial weight configuration, the absolute differences are on the order of $10^{-13}$ to $10^{-14}$, supporting the correctness of the implemented analytical derivatives.

## Scope and Limitations

The computational experiments are designed to illustrate the mathematical behaviour of forward propagation, backpropagation, and gradient-based optimisation rather than evaluate the predictive performance of neural networks on real-world datasets.

Several limitations should be considered:

- The network has a small, fixed 2–2–1 architecture.
- Bias terms are omitted from the numerical implementation.
- Only one input-target pair is used.
- All learning-rate experiments begin from the same initial weight configuration.
- The experiments use full-batch gradient descent on a single example.
- The results do not establish general convergence behaviour across different network architectures, datasets, or initialisations.

The findings should therefore be interpreted as a controlled numerical illustration of the mathematical principles discussed in the paper.

## Citation

If you use this repository or its accompanying mathematical exposition, please cite:

Ahmadi, M. E. (2026). *A Mathematical Exposition of Forward Propagation and Backpropagation in Artificial Neural Networks*. Version 1.0. Independent mathematical exposition.

A permanent archival DOI may be added after the manuscript is deposited in a suitable research repository.

## Code Availability

The Python implementations, experimental figures, and numerical results are publicly available in this repository:

https://github.com/ely4s-4hmadi/ann-forward-propagation-backpropagation

The original implementation is provided for mathematical clarity, while the extended implementation is recommended for computational reproducibility.
