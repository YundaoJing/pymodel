# Readable KAN Visualization

This repository provides an improved visualization interface for Kolmogorov-Arnold Networks (KANs), with a particular focus on readability for large and structurally complex KAN models.

The code is developed based on the original KAN implementation and is intended to make the model structure more explicit, interpretable, and convenient to inspect in scientific machine-learning workflows.

Original KAN repository:

- https://github.com/KindXiaoming/pykan

Original KAN paper:

- Liu, Z. et al. KAN: Kolmogorov-Arnold Networks. International Conference on Learning Representations (ICLR), 2025.
- arXiv: https://arxiv.org/abs/2404.19756
- OpenReview: https://openreview.net/forum?id=Ozo7qJ5vZi

## Overview

Kolmogorov-Arnold Networks (KANs) replace fixed node-wise activation functions in conventional multilayer perceptrons with learnable edge-wise activation functions. This design makes KANs attractive for scientific applications where model interpretability, symbolic analysis, and visualization are important.

However, when the network becomes large, the default visualization can become difficult to read because many spline functions, connections, and neuron-wise components are displayed simultaneously. This repository improves the display layer of KAN models by explicitly representing the network structure and by making large-network visualization more readable.

## Main Features

- Improved visualization of KAN architectures, especially for large networks.
- More explicit rendering of KAN structures, including layer-wise and edge-wise components.
- Cleaner display layout for complex networks.
- Better readability for scientific figures, model inspection, and supplementary materials.
- Compatible with workflows based on the original `pykan` implementation.

## Relationship to the Original KAN Code

This repository is not a replacement for the original KAN project. Instead, it provides modifications and extensions mainly targeting the model display and visualization components.

The original KAN implementation is available at:

```text
https://github.com/KindXiaoming/pykan
