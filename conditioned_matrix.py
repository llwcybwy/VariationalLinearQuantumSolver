import numpy as np

def create_conditioned_random_matrix(qubits: int, condition_number: float):
    """
    Generates a random qubit x qubit-matrix with the given condition number.

    :param qubits: Size of matrix.
    :param condition_number: Condition number of matrix.
        
    """
    # Generate complex matrix
    m_real_unconditioned = np.random.random((qubits, qubits))
    m_imag_unconditioned = np.random.random((qubits, qubits)) * 1.j
    m_unconditioned = m_imag_unconditioned + m_real_unconditioned
    U, S_unconditioned, Vh = np.linalg.svd(m_unconditioned)
    sigma_max = S_unconditioned[0]; sigma_min = sigma_max/condition_number
    # Create conditioned singular values
    diag = np.random.uniform(low=sigma_min, high=sigma_max, size=(qubits,))
    diag[::-1].sort()
    diag[0] = sigma_max; diag[-1] = sigma_min
    S_conditioned=np.diag(diag)
    # Reassemble matrix
    m_conditioned = U @ S_conditioned @ Vh
    return m_conditioned

if __name__ == "__main__":
    m = create_conditioned_random_matrix(100, 1000)
    print(f"condition number is {np.linalg.cond(m)}")