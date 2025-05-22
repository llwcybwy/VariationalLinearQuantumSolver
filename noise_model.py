import pennylane as qml
import numpy as np

# @qml.BooleanFn
# def rot_cond(op):
#     return isinstance(op, qml.Rot) 

# depol_error = qml.noise.partial_wires(qml.DepolarizingChannel, 0.0)

# fcond_rot, noise_rot = rot_cond, depol_error

# @qml.BooleanFn
# def CZ_cond(op):
#     return isinstance(op, qml.CZ) 

# depol_error = qml.noise.partial_wires(qml.DepolarizingChannel, 0.0)

# fcond_rot, noise_rot = CZ_cond, depol_error

# noise_model = qml.NoiseModel(
#     {fcond_rot: noise_rot, CZ_cond: depol_error}, 
# )

import pennylane as qml

# Condition 1: apply noise to Rot gates
@qml.BooleanFn
def rot_cond(op):
    return isinstance(op, qml.Rot)

rot_noise = qml.noise.partial_wires(qml.DepolarizingChannel, 0.05)  # e.g. 5% noise

# Condition 2: apply noise to CZ gates
@qml.BooleanFn
def cz_cond(op):
    return isinstance(op, qml.CZ)

cz_noise = qml.noise.partial_wires(qml.DepolarizingChannel, 0.1)  # different noise level

# Combine into a NoiseModel
noise_model = qml.NoiseModel(
    model_map={
        rot_cond: rot_noise,
        cz_cond: cz_noise,
    }
)

