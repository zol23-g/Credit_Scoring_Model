from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained models
with open('models/log_reg_model.pkl', 'rb') as f:
    log_reg_model = pickle.load(f)

with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/gbm_model.pkl', 'rb') as f:
    gbm_model = pickle.load(f)


# Sample input for testing
def create_input_df(data):
    """
    Convert the JSON input into a pandas DataFrame for model prediction.
    """
    input_df = pd.DataFrame([data], columns=['Recency', 'Frequency', 'Monetary', 'Size'])
    return input_df


# Root endpoint
@app.route('/')
def home():
    return "Credit Scoring Prediction API is up and running!"

# Logistic Regression Prediction Endpoint
@app.route('/predict/logistic_regression', methods=['POST'])
def predict_log_reg():
    data = request.get_json(force=True)
    input_df = create_input_df(data)
    
    # Make predictions
    prediction = log_reg_model.predict(input_df)
    probability = log_reg_model.predict_proba(input_df)[:, 1]
    
    return jsonify({
        'model': 'Logistic Regression',
        'prediction': int(prediction[0]),  # 0 for bad, 1 for good
        'probability': float(probability[0])  # Probability of being good
    })

# Random Forest Prediction Endpoint
@app.route('/predict/random_forest', methods=['POST'])
def predict_rf():
    data = request.get_json(force=True)
    input_df = create_input_df(data)
    
    prediction = rf_model.predict(input_df)
    probability = rf_model.predict_proba(input_df)[:, 1]
    
    return jsonify({
        'model': 'Random Forest',
        'prediction': int(prediction[0]),
        'probability': float(probability[0])
    })

# Gradient Boosting Prediction Endpoint
@app.route('/predict/gradient_boosting', methods=['POST'])
def predict_gbm():
    data = request.get_json(force=True)
    input_df = create_input_df(data)
    
    prediction = gbm_model.predict(input_df)
    probability = gbm_model.predict_proba(input_df)[:, 1]
    
    return jsonify({
        'model': 'Gradient Boosting',
        'prediction': int(prediction[0]),
        'probability': float(probability[0])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
