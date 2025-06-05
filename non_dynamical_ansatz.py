import pennylane as qml

def fixed_layered_ansatz(params):
    wires = range(4)  
    depth = 10
    n_qubits = len(wires)
    param_idx = 0

    for i in range(n_qubits):
            qml.RY(params[param_idx], wires=wires[i])
            param_idx += 1

    for d in range(depth):
        # CZ gates on alternating pairs 
        for i in range(0, n_qubits - 1, 2):
            qml.CZ(wires=[wires[i], wires[i + 1]])
            
        # RY rotations on all 
        for i in range(n_qubits):
            qml.RY(params[param_idx], wires=wires[i])
            param_idx += 1

        # CZ gates on alternating pairs (odd pairs)
        for i in range(1, n_qubits - 2, 2):
            qml.CZ(wires=[wires[i], wires[i + 1]])

        # RY rotations on all qubits, except the first and last one
        for i in range(1, n_qubits-1):
            qml.RY(params[param_idx], wires=wires[i])
            param_idx += 1
