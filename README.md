# Dissipative Hamiltonian Neural Networks
Andrew Sosanya, Sam Greydanus | 2020

**Note: the name of this project was recently changed from "Deep Helmholtz Decompositions" to "Dissipative Hamiltonian Neural Networks." So if you see a reference to the former, mentally replace it for the latter. We aim to correct any discrepacies in future releases.**

* Paper: [https://arxiv.org/abs/2201.10085](https://arxiv.org/abs/2201.10085)
* Blog: [greydanus.github.io/2021/06/11/dissipative-hnns/](https://greydanus.github.io/2021/06/11/dissipative-hnns/)

![Main idea](./static/hero.jpg)
_Dissipative HNNs (D-HNNs) output two scalar functions, denoted here with **H** and **D**. The first of these two, **H**, is the Hamiltonian. It is perfectly conserved. The second of these two, **D**, is the Rayleigh dissipation function. It models the dissipative component of the dynamics of a physical system. The addition of the dissipation function is what sets this model apart from Hamiltonian Neural Networks; it allows D-HNNs to model systems where energy is not quite conserved, as, for example, in the case of a damped pendulum._

Basic usage
--------

Use the .ipnyb notebooks to train and analyze all models

Summary
--------

We propose a simple way of extending Hamiltonian Neural Networks so as to model physical systems with dissipative forces. We call this model a Dissipative Hamiltonian Neural Network (D-HNN) because it adds support for dissipative dynamics.

Dependencies
--------
 * PyTorch
 * NumPy
 * ImageIO
 * Scipy
 
This project is written in Python 3.