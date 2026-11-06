"""
Diabetes Prediction System - Flask web app
Works locally (python app.py) and on Render (gunicorn app:app)
"""

import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

model = None
scaler = None

# Field names must match the ones used in static/js/script.js
FEATURES_8 = [
    'pregnancies', 'glucose', 'bloodPressure', 'skinThickness',
    'insulin', 'bmi', 'diabetesPedigree', 'age',
]
FEATURES_7 = FEATURES_8[1:]  # same list without pregnancies


def load_model():
    """Load the trained model and scaler from disk."""
    global model, scaler

    # Prefer the improved (7-feature) model if it exists, else the original
    pairs = [
        ('model_improved.pkl', 'scaler_improved.pkl'),
        ('model.pkl', 'scaler.pkl'),
    ]
    for model_file, scaler_file in pairs:
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            try:
                model = joblib.load(model_file)
                scaler = joblib.load(scaler_file)
                print(f"Model loaded: {model_file}")
                return True
            except Exception as e:
                print(f"Error loading {model_file}: {e}")

    print("Model files not found. Run train_model.py first.")
    return False


def get_risk_level(probability):
    if probability < 0.3:
        return 'Low'
    elif probability < 0.6:
        return 'Moderate'
    return 'High'


def get_message(prediction, probability):
    pct = probability * 100
    if prediction == 0:
        return (f"Low risk of diabetes detected ({pct:.1f}% probability). "
                "Maintain a healthy lifestyle.")
    return (f"High risk of diabetes detected ({pct:.1f}% probability). "
            "Please consult a healthcare professional.")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not trained. Please run train_model.py first.'}), 400

    try:
        data = request.get_json()

        # Use 7 or 8 inputs depending on which model was trained
        names = FEATURES_8 if scaler.n_features_in_ == 8 else FEATURES_7
        features = [float(data[name]) for name in names]

        features_scaled = scaler.transform(np.array(features).reshape(1, -1))
        prediction = int(model.predict(features_scaled)[0])
        probability = model.predict_proba(features_scaled)[0]

        return jsonify({
            'prediction': prediction,
            'probability': {
                'no_diabetes': float(probability[0]),
                'diabetes': float(probability[1]),
            },
            'risk_level': get_risk_level(probability[1]),
            'message': get_message(prediction, probability[1]),
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/metrics')
def metrics():
    return jsonify({
        'status': 'Model loaded and ready' if model is not None else 'Model not loaded',
        'model_type': type(model).__name__ if model is not None else None,
        'features_used': int(scaler.n_features_in_) if scaler is not None else None,
    })


# Runs when gunicorn imports the app (Render) AND when running python app.py
load_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
