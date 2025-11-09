import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

# ---------- Load Data ----------
print("📦 Loading TF-IDF matrix and DataFrame...")
data = joblib.load("policy_tfidf_matrix.pkl")
df = data["df"]
X = data["matrix"].toarray()

# ---------- Prepare Labels ----------
print("🔍 Preparing binary labels from 'status' column...")
y = df["status"].apply(lambda x: 1 if str(x).lower() == "active" else 0)

# ---------- Normalize + Reduce Dimensions ----------
print("📊 Scaling and reducing dimensions with PCA...")
X = StandardScaler().fit_transform(X)
n_qubits = 4
pca = PCA(n_components=n_qubits)
X = pca.fit_transform(X)

# ---------- Train/Test Split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
y_train = np.array(y_train)
y_test = np.array(y_test)

# ---------- Quantum Circuit ----------
print("⚛️ Initializing quantum device and circuit...")
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

def cost(weights, X, y):
    predictions = []
    for x in X:
        x = np.array(x, dtype=float)
        predictions.append(circuit(weights, x=x))
    return np.mean((np.array(predictions) - y) ** 2)

# ---------- Training Loop ----------
print("🧪 Starting training...")
weights = np.random.randn(n_qubits, requires_grad=True)
opt = AdamOptimizer(0.1)

for i in range(100):
    weights = opt.step(lambda w: cost(w, X_train, y_train), weights)
    if i % 10 == 0:
        print(f"✅ Step {i} complete")

# ---------- Save Model ----------
print("💾 Saving trained weights to quantum_weights.pkl...")
joblib.dump(weights, "quantum_weights.pkl")
print("🎉 Training complete and weights saved.")