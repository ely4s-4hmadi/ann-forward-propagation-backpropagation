# Artificial Neural Networks: Forward Propagation and Backpropagation

This repository contains the Python code used for the computational experiment in my paper, *A Mathematical Exposition of Forward Propagation and Backpropagation in Artificial Neural Networks*.

The experiment explores how a small feedforward neural network learns through forward propagation, backpropagation and gradient descent.

## Contents

- `ComputationalExperiment.py` — implements the neural network and compares different learning rates.

## Requirements

- Python 3
- NumPy
- Matplotlib

## Installation

Install the required libraries using:

```bash
py -m pip install -r requirements.txt
```

## Running the Experiment

Run the following command from the repository directory:

```bash
py ComputationalExperiment.py
```

The script performs the computational experiment using four different learning rates:

- 0.01
- 0.1
- 0.3
- 0.8

It tracks the loss over successive training iterations and produces a comparison of the learning behaviour for each learning rate.

## Reproducibility

The code is provided to allow readers to reproduce the computational experiment and explore how different learning rates affect the training process.

Readers can also modify the initial parameters and experiment settings to investigate their effects on the network's behaviour.

## Author

Elyas Ahmadi  
Computer Science, University of Liverpool
