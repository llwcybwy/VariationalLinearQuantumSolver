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
from device import Device

qml.QubitStateVector = qml.StatePrep


class DeviceVarLSESolver(VarLSESolver):
    """
    This class implements a variational LSE solver with customizable loss functions.
    """

    def __init__(
            self,
            a: np.ndarray | list[str | np.ndarray | Callable],
            b: np.ndarray | Callable,
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
            device: Device = None
    ):
        self.device = device

        VarLSESolver.__init__(self,a,b,coeffs,ansatz,weights,method,local,lr,steps,epochs,threshold,abort,seed,data_qubits)

    def qnode_evaluate_x(self) -> Callable:
        """
        Quantum node that evaluate V(alpha)

        :return: Circuit handle evaluating V(alpha)
        """

        dev = self.device.getDevice()

        @qml.qnode(dev, interface="torch")
        def circuit_evolve_x(weights):
            """
            Circuit that evaluates V(alpha)

            :param weights: Parameters for the VQC.
            """
            self.ansatz(weights)
            
            return qml.probs()

        return circuit_evolve_x