import numpy as np
from scipy.sparse import hstack
import pickle

# Load model
with open("srd_mrd_classifier.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Prediction function
def predict_query(query):
    vec = vectorizer.transform([query])
    
    # length feature (IMPORTANT if used in training)
    length = np.array([len(query.split())]).reshape(-1,1)
    
    final = hstack([vec, length])
    
    pred = model.predict(final)[0]
    return pred


# ===============================
# USER INPUT LOOP
# ===============================
while True:
    query = input("\nEnter your query (type 'exit' to stop): ")
    
    if query.lower() == "exit":
        print("Stopped.")
        break
    
    result = predict_query(query)
    
    print("👉 Prediction:", result)