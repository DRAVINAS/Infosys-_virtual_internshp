import joblib
import pennylane as qml
from pennylane import numpy as np

weights = joblib.load("quantum_weights.pkl")
n_qubits = weights.shape[0]  

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(weights, x=None):
    x = np.array(x, dtype=float)
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    return qml.expval(qml.PauliZ(0))

def predict_score(x_vec):
    return circuit(weights, x=x_vec)