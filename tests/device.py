import pennylane as qml
from enum import Enum

class DeviceType(Enum):
    DEFAULT = 'DEFAULT'
    IBM = 'IBM'
    QISKIT_AER = 'QISKIT_AER'

class Device:
    def __init__(self,
                 type: DeviceType,
                 qubits: int = None,
                 backend = None):
        self.type = type
        self.qubits = qubits
        self.backend = backend
    
    def getDevice(self):
        match self.type:
            case DeviceType.IBM:
                try:
                    assert not self.backend is None
                    return qml.device("qiskit.remote", wires=self.qubits, backend=self.backend)
                except Exception as e:
                    print(e)
            case DeviceType.QISKIT_AER:
                return qml.device('qiskit.aer', wires=self.qubits)
            case DeviceType.DEFAULT:
                return qml.device('default.qubit', wires=self.qubits)
            case '':
                print("Invalid type argument.")
                return ValueError
            
if __name__ == "__main__":
    dev = Device(DeviceType.QISKIT_AER)
    print(dev)
    print(dev.getDevice())