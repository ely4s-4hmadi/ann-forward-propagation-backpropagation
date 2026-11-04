# A Mathematical Exposition of Forward Propagation and Backpropagation in Artificial Neural Networks

**Author:** Mohammad Elyas Ahmadi  
**Affiliation:** School of Computer Science and Informatics, University of Liverpool  
**Version:** 1.0 — November 2026

## Overview

This repository accompanies the mathematical exposition *A Mathematical Exposition of Forward Propagation and Backpropagation in Artificial Neural Networks*.

It provides a reproducible Python implementation of a feedforward neural network with two input neurons, two hidden neurons and one output neuron.

The implementation demonstrates forward propagation, analytical gradient calculation through backpropagation, gradient descent, numerical gradient verification using central finite differences, and the effects of different learning rates.

## Experimental configuration

- Network architecture: 2–2–1
- Activation function: Sigmoid
- Loss function: Half squared error
- Inputs: 0.35 and 0.9
- Target output: 0.8
- Initial weights: [0.1, 0.4, 0.8, 0.6, 0.3, 0.9]
- Learning rates: 0.01, 0.1, 0.3 and 0.8
- Training iterations: 1,000
- Numerical gradient-checking step: 0.00001

Biases are omitted from the numerical network to keep the mathematical derivations accessible.

## Requirements

Python 3 with NumPy and Matplotlib.

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Reproducing the experiments

Run:

```bash
python ComputationalExperiment.py
```

The script calculates the numerical example, verifies the analytical gradients, compares the four learning rates, and saves the results and figures to the `ann_results` directory.

## Generated outputs

- **Figure 12:** Loss before and after one gradient-descent update.
- **Figure 13:** Loss trajectories for four learning rates.
- **Figure 14:** Absolute prediction error after training.
- **gradient_check.csv:** Comparison of analytical and numerical gradients.
- **learning_rate_summary.csv:** Final outputs and losses for each learning rate.
- **loss_trajectories.csv:** Loss recorded throughout training.
- **single_step.csv:** Numerical results for the single-step update.
- **experiment_metadata.json:** Experimental configuration and software information.

## Scope and limitations

The computational experiment uses one fixed input-target pair and a small neural network without biases. It is intended to illustrate the mathematical behaviour of gradient-based learning rather than establish general performance across datasets or architectures.

## Citation

Ahmadi, M. E. (2026). *A Mathematical Exposition of Forward Propagation and Backpropagation in Artificial Neural Networks*. Version 1.0. Independent mathematical exposition.

## Code availability

Repository: https://github.com/ely4s-4hmadi/ann-forward-propagation-backpropagation
