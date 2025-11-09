import joblib
from quantum_model import predict_score

# Load vectorizer
vectorizer = joblib.load("policy_vectorizer.pkl")

# Sample query
query = "digital education access in rural India"
query_vec = vectorizer.transform([query.lower()]).toarray()[0]

# Predict quantum score
score = predict_score(query_vec)
print(f"Quantum score for query: {score:.3f}")