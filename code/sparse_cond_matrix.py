import numpy as np
import pennylane as qml

# Define single-qubit Pauli matrices
I_ = np.array([[1.0, 0.0], [0.0, 1.0]])
X_ = np.array([[0.0, 1.0], [1.0, 0.0]])
Y_ = np.array([[0.0, -1j], [1j, 0.0]])
Z_ = np.array([[1.0, 0.0], [0.0, -1.0]])

pauli_dict = {'I': I_, 'X': X_, 'Y': Y_, 'Z': Z_}


def pauli_string_to_matrix(pauli_string):
    """Convert a Pauli string like 'XZI' to a matrix via Kronecker products."""
    mats = [pauli_dict[p] for p in pauli_string]
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def build_matrix_from_paulis(pauli_strings, coeffs):
    """Reconstruct the dense matrix A from a list of Pauli strings and coefficients."""
    A = np.zeros((2**len(pauli_strings[0]), 2**len(pauli_strings[0])), dtype=complex)
    for ps, c in zip(pauli_strings, coeffs):
        A += c * pauli_string_to_matrix(ps)
    return A


def build_pauli_sum_A_strings(qubits, J=0.1, kappa=10):
    """Construct A and b from Eq. (26) in Bravo-Prieto et al., 2023
    Returns A as Pauli strings and coefficients, and |b⟩ = H^{⊗n}|0⟩
    """

    # Build Hamiltonian: sum_j X_j + J * sum_j Z_j Z_{j+1}
    coeffs = []
    pauli_strings = []

    # Add X terms: X_j
    for i in range(qubits):
        ps = ['I'] * qubits
        ps[i] = 'X'
        pauli_strings.append(''.join(ps))
        coeffs.append(1.0)

    # Add ZZ terms: Z_j Z_{j+1}
    for i in range(qubits - 1):
        ps = ['I'] * qubits
        ps[i] = 'Z'
        ps[i + 1] = 'Z'
        pauli_strings.append(''.join(ps))
        coeffs.append(J)

    # Build PennyLane ops for eigenvalue estimation
    ops = []
    for ps in pauli_strings:
        term = None
        for j, p in enumerate(ps):
            op = {
                'I': qml.Identity(j),
                'X': qml.PauliX(j),
                'Y': qml.PauliY(j),
                'Z': qml.PauliZ(j)
            }[p]
            term = op if term is None else term @ op
        ops.append(term)

    H_raw = qml.Hamiltonian(coeffs, ops)
    Hmat = H_raw.sparse_matrix().toarray()
    eigvals = np.linalg.eigvalsh(Hmat)
    E_min = np.min(eigvals)
    E_max = np.max(eigvals)

    # Rescale A to get desired condition number
    eta = (E_max - kappa * E_min) / (kappa - 1)
    xi = E_max + eta

    # Rescale coefficients
    coeffs = [c / xi for c in coeffs]

    # Add eta * I term
    coeffs.append(eta / xi)
    pauli_strings.append('I' * qubits)

    # Create |b⟩ = H^{⊗n}|0⟩ = uniform superposition
    b_state = np.ones(2**qubits) / np.sqrt(2**qubits)

    return pauli_strings, coeffs, b_state

