All files in this folder contain a Dictionary as follows.

saved_dict = {
    # Noise dynamic
    "local_noise_dynamic_trc" : local_noise_dynamic_trc,
    "local_noise_dynamic_error" : local_noise_dynamic_error,
    "local_noise_dynamic_solution" : local_noise_dynamic_solution,
    "global_noise_dynamic_trc" : global_noise_dynamic_trc,
    "global_noise_dynamic_error" : global_noise_dynamic_error,
    "global_noise_dynamic_solution" : global_noise_dynamic_solution,

    # Noiseless dynamic
    "local_noiseless_dynamic_trc" : local_noiseless_dynamic_trc,
    "local_noiseless_dynamic_error" : local_noiseless_dynamic_error,
    "local_noiseless_dynamic_solution" : local_noiseless_dynamic_solution,
    "global_noiseless_dynamic_trc" : global_noiseless_dynamic_trc,
    "global_noiseless_dynamic_error" : global_noiseless_dynamic_error,
    "global_noiseless_dynamic_solution" : global_noiseless_dynamic_solution,

    # Noise static
    "local_noise_static_trc" : local_noise_static_trc,
    "local_noise_static_error" : local_noise_static_error,
    "local_noise_static_solution" : local_noise_static_solution,
    "global_noise_static_trc" : global_noise_static_trc,
    "global_noise_static_error" : global_noise_static_error,
    "global_noise_static_solution" : global_noise_static_solution,

    # Noiseless static
    "local_noiseless_static_trc" : local_noiseless_static_trc,
    "local_noiseless_static_error" : local_noiseless_static_error,
    "local_noiseless_static_solution" : local_noiseless_static_solution,
    "global_noiseless_static_trc" : global_noiseless_static_trc,
    "global_noiseless_static_error" : global_noiseless_static_error,
    "global_noiseless_static_solution" : global_noiseless_static_solution,

    "normalized_classical_solution" : normalized_classical_solution
}

The values have the following shapes.

# Dense
np.matrix((redo_calc, len(cs))) for all but the solutions which have the shape np.matrix((redo_calc, len(cs), 2**qubits)).

`redo_calc`=5 for all files. 
For dense results 1 and 2, cs = [1.04e+00, 1.49e+00, 1.58e+02] (in effect giving one file with redo_calc=10)
For dense results 3, cs = [10, 50, 100]

# Poisson
np.matrix((redo_calc, )) for all but the solutions which have the shape np.matrix((redo_calc, 2**qubits)) 
For Poisson results 1 and 2, `redo_calc`=5 (in effect giving one file with redo_calc=10)