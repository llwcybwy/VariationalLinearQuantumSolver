# Variational Linear Quantum Solver

Written by Joakim Colpier and Anna Ingmer Boye.

This project is based by the VLQS implementation by [article by Meyer et al.](https://arxiv.org/abs/2404.09916). Our project consists of recreating the results provided by the paper, and study what effect quantum noise has on solver. Acces the GitHub for the project using [the following link](link to project).

## To get started

The packages to download along with their versions are as follows.
`python 3.12`
`numpy~=1.26.4` (**not** version 2 or higher)
`torch~=2.2.2`
`pennylane~=0.41.1`
`variational-lse-solver~=1.0`
`pennylane-qiskit~=0.41.0.post0`
`qiskit-aer~=0.16.0`
`matplotlib~=3.10.3`


## Folder overview

This section will contain an overview of the content of the GitHub page.

### Code

Contains the runnable code for the project, including tests, recreation of Meyer et al.'s results, as well as custom classes and functions. Code which is not to be ran have the python file format, whereas tests and recreation of results (files that should be ran) have the file format Jupyter Notebook. 

### Presentation

Presented for the course WI4650 Quantum Algorithms, 16 June 2025 at TU Delft, detailing our findings and process.

### Texts

Most relevant texts provided with the course project.

## Acknowledgements

- The project mainly bases itself on the works of Bravo-Prieto et al., Patil et al. and Meyer et al. (see presentation for correct referrencing and references to other works that have been used to a lesser extent).
- The code is mainly adapted from Meyre et al. ([link to GitHub](https://github.com/nicomeyer96/variational-lse-solver)).
- The presentation template used for the presentation was created by Erwin Walraven and Maarten Abbink.
