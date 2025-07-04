import pennylane as qml
from enum import Enum
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import GenericBackendV2

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
                backend = GenericBackendV2(num_qubits=self.qubits)
                noise_model = NoiseModel.from_backend(backend)
                return qml.device('qiskit.aer', wires=self.qubits, noise_model=noise_model)
            case DeviceType.DEFAULT:
                return qml.device('default.qubit', wires=self.qubits)
            case '':
                print("Invalid type argument.")
                return ValueError
            
if __name__ == "__main__":
    dev = Device(DeviceType.QISKIT_AER)
    print(dev)
    print(dev.getDevice())