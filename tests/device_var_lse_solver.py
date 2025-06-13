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
from device import Device, DeviceType
import torch
import sys
from tqdm import tqdm

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
            device: Device = None,
            silent: bool = False
    ):
        self.silent = silent
        self.device = device
        if self.device == None:
            self.device = Device(DeviceType.DEFAULT, qubits=data_qubits)

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
    
    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the LSE provided during initialization.

        :return: Solution of the LSE (proportional), Associated variational parameters, Iteration count (epoch -> step amount)
        """

        # current best weights (i.e. the ones producing the lowest loss)
        best_weights = self.weights.detach().numpy()
        iteration_count = {}

        for epoch in range(self.epochs):

            # best loss with corresponding step it was achieved in during this epoch
            best_loss, best_step = 1.0, 0

            # append additional layer to dynamic circuit (skip for first epoch) and re-register optimizer
            if 0 < epoch:
                if not self.silent: print('Increasing circuit depth.', flush=True)
                new_weights = np.random.uniform(low=0.0, high=2 * np.pi, size=(1, self.weights.shape[1]))
                weights = np.concatenate((best_weights, np.stack((new_weights,
                                                                  np.zeros((1, self.weights.shape[1])),
                                                                  -new_weights), axis=2)))
                self.weights = torch.tensor(weights, requires_grad=True)
                self.opt = torch.optim.Adam([{'params': self.weights}], lr=self.lr)

            # train until either maximum number of steps is reached, early stopping criteria is fulfilled,
            # or no loss function change in several consecutive steps (increase depth in this case)
            if self.silent:
                pbar = range(self.steps)
            else:
                pbar = tqdm(range(self.steps), desc=f'Epoch {epoch+1}/{self.epochs}: ', file=sys.stdout)
            for step in pbar:
                iteration_count[epoch+1] = step+1
                self.opt.zero_grad()
                # compute loss
                loss = self.cost_function.cost(self.weights)
                # test is loss has improved beyond 0.1 * `threshold`
                # (ensures increasing depth when only negligible improvements are made)
                if loss.item() < best_loss and abs(loss.item() - best_loss) > 0.1 * self.threshold:
                    best_loss = loss.item()
                    best_step = step
                    best_weights = self.weights.detach().numpy()
                # test if stopping threshold has been reached
                if loss.item() < self.threshold:
                    if not self.silent: 
                        pbar.close()
                        print(f'Loss of {loss.item():.10f} below stopping threshold.\nReturning solution.', flush=True)
                    return self.evaluate(best_weights), best_weights, iteration_count
                # if loss has not improved in the last `abort` steps terminate this epoch and increase depth
                if step - best_step >= self.abort:
                    if not self.silent: 
                        pbar.close()
                        print(f'Loss has not improved in last {self.abort} steps.', flush=True) \
                        if epoch < self.epochs - 1 \
                        else print(f'Loss has not improved in last {self.abort} steps.\nReturning best solution.', flush=True)
                    break
                # log current loss to progress bar
                if not self.silent: pbar.set_postfix({'best loss': best_loss, 'last improvement in step': best_step, 'loss': loss.item()})
                # determine gradients and update
                loss.backward()
                self.opt.step()
        return self.evaluate(best_weights), best_weights, iteration_count