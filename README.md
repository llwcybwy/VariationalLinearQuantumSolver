# Variational Linear Quantum Solver

This project is based by the VLQS implementation by [article by Meyer et al.](https://arxiv.org/abs/2404.09916). Our project consists of recreating the results provided by the paper, and study what effect quantum noise has on solver.

## To get started

The packages to download along with their versions are as follows.
`python 3.12`
`numpy~=1.26.4` (**not** version 2 or higher)
`torch~=2.2.2`
`pennylane~=0.41.1`
`variational-lse-solver~=1.0`
`pennylane-qiskit~=0.41.0.post0`
`qiskit-aer~=0.16.0`


## Overview of the project

This is where we explain what the different files do. 

### NoiseVarLSESolver

Takes in a custom noise model which is applied to the circuit directly.

### DeviceVarLSESolver

Takes a device that the circuit is run upon, thus indirectly implementing noise.
