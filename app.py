from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests from the frontend

# Dummy training data
X = np.array([
    [1, 1, 1, 1],
    [0, 1, 1, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 0],
])
y = np.array([1, 1, 0, 0, 1, 0])

model = LogisticRegression()
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = [
        int(data.get("fever", 0)),
        int(data.get("dry_cough", 0)),
        int(data.get("fatigue", 0)),
        int(data.get("breathing", 0)),
    ]
    prediction = model.predict([symptoms])[0]
    response = "You might have COVID-19. Please consult a doctor." if prediction else "You are unlikely to have COVID-19."
    return jsonify({"prediction": response})

if __name__ == '__main__':
    app.run(debug=True)
