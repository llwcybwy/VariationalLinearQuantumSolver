import pennylane as qml
import numpy as np

@qml.BooleanFn
def rot_cond(op):
    return isinstance(op, qml.Rot) 

depol_error = qml.noise.partial_wires(qml.DepolarizingChannel, 0.0)

fcond_rot, noise_rot = rot_cond, depol_error

@qml.BooleanFn
def CZ_cond(op):
    return isinstance(op, qml.CZ) 

depol_error = qml.noise.partial_wires(qml.DepolarizingChannel, 0.0)

fcond_rot, noise_rot = CZ_cond, depol_error

noise_model = qml.NoiseModel(
    {fcond_rot: noise_rot, CZ_cond: depol_error}, 
)

