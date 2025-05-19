# This code is part of the variational-lse-solver library.
#
# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This file contains the main access point for using the variational LSE solver.
"""

import numpy as np
import pennylane as qml
from typing import Callable
from variational_lse_solver import VarLSESolver
from noise_model import noise_model


class NoisyVarLSESolver(VarLSESolver):
    """
    This class implements a variational LSE solver with customizable loss functions and noise.
    """

    def __init__(
            self,
            a: np.ndarray | list[str | np.ndarray | Callable],
            b: np.ndarray | Callable,
            noise_model: qml.NoiseModel = None,
            coeffs: list[float | complex] = None,
            ansatz: Callable = None,
            weights: tuple[int, ...] | np.ndarray = None,
            method: str = 'direct',
            local: bool = False,
            lr: float = 0.01,
            steps: int = 10000,
            epochs: int = 1,
            threshold: float = 1e-4,
            abort: int = 500,
            seed: int = None,
            data_qubits: int = 0,
    ):
        self.noise_model=noise_model
        VarLSESolver.__init__(self,a,b,coeffs,ansatz,weights,method,local,lr,steps,epochs,threshold,abort,seed,data_qubits)

    def evaluate(self, weights: np.array) -> np.array:
        """
        Return measurement probabilities for the state prepared as solution of the LSE.

        :param weights: Weights for the VQC ansatz.
        :return: Measurement probabilities for the state V(alpha)
        """
        return self.qnode_evaluate_x()(weights).detach().numpy()
    
    def qnode_evaluate_x(self) -> Callable:
        """
        Quantum node that evaluate V(alpha)

        :return: Circuit handle evaluating V(alpha)
        """
        dev = qml.device('default.mixed', wires=self.data_qubits)

        def circuit_evolve_x(weights):
            """
            Circuit that evaluates V(alpha)

            :param weights: Parameters for the VQC.
            """
            self.ansatz(weights)
            circuit_evolve_x = qml.qnode(dev, interface='torch')(qml.probs())
            noisy_circuit_evolve_x = qml.add_noise(circuit_evolve_x, self.noise_model)
            return noisy_circuit_evolve_x

        return circuit_evolve_x
    
# Test
if __name__ == "__main__":
    I_ = np.array([[1.0, 0.0], [0.0, 1.0]])
    X_ = np.array([[0.0, 1.0], [1.0, 0.0]])
    Y_ = np.array([[0.0, -1.j], [1.j, 0.0]])
    Z_ = np.array([[1.0, 0.0], [0.0, -1.0]])

    a = ["III", "XZI", "XII"]
    b = np.ones(8)/np.sqrt(8)

    NoisyVarLSESolver(a,
                      b,
                      noise_model,
                      coeffs=[1.0, 0.2, 0.2], 
                      method="hadamard", 
                      local=True, 
                      lr=0.1, 
                      steps=50,
                      threshold=0.001, 
                      epochs=10)