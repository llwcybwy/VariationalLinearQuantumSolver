import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

def conditionedMatrix(qubits: int, condition_number: float):
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

def poissonMatrix(n: int):
    """
    Generates the discrete Poisson equation matrix for an n x n grid (2D Laplacian with Dirichlet boundary conditions).
    :param n: Number of grid points in one dimension.
    :return: (n*n, n*n) scipy.sparse.csr_matrix representing the Poisson matrix.
    """
    N = n * n
    main_diag = 4 * np.ones(N)
    side_diag = -1 * np.ones(N - 1)
    side_diag[np.arange(1, N) % n == 0] = 0  # zero out wrap-around connections
    up_down_diag = -1 * np.ones(N - n)

    diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
    offsets = [0, -1, 1, -n, n]
    A = sp.diags(diagonals, offsets, shape=(N, N), format='csr')
    return A

if __name__ == "__main__":
    m = conditionedMatrix(100, 1000)
    print(f"condition number for dense is {np.linalg.cond(m)}")
    m_p = poissonMatrix(4)
    print(m_p)
    print(m_p.todense())
    # print(f"condition number for Poisson is {np.linalg.cond(m_p)}")