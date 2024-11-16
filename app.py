from flask import Flask, request, jsonify
import pickle
import pandas as pd
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained Logistic Regression model
with open('models/log_reg_model.pkl', 'rb') as f:
    log_reg_model = pickle.load(f)

def calculate_rfms(transactions):
    """
    Calculate Recency, Frequency, Monetary, and Size based on multiple transaction data.
    """
    df = pd.DataFrame(transactions)
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

    # Convert 'Amount' to numeric, coercing errors to NaN
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Drop rows where 'Amount' could not be converted to a number
    df = df.dropna(subset=['Amount'])

    # Check if there are any rows left after dropping NaN values
    if df.empty:
        raise ValueError("No valid transaction data available after processing. Check 'Amount' values.")

    # Calculate RFMS features for each customer
    recency = df.groupby('CustomerId')['TransactionDate'].max().apply(lambda x: (datetime.now() - x).days)
    frequency = df.groupby('CustomerId')['TransactionId'].count()
    monetary = df.groupby('CustomerId')['Amount'].sum()
    size = df.groupby('CustomerId')['Amount'].mean()

    rfms_df = pd.DataFrame({
        'Recency': recency,
        'Frequency': frequency,
        'Monetary': monetary,
        'Size': size
    }).reset_index()

    return rfms_df

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Ensure data is in the expected format (list of transactions)
    if not isinstance(data, list):
        return jsonify({'error': 'Input data must be a list of transactions'}), 400
    
    try:
        # Calculate RFMS features for all customers
        rfms_df = calculate_rfms(data)

        # Prepare response for each customer
        results = []
        for _, row in rfms_df.iterrows():
            input_df = row[['Recency', 'Frequency', 'Monetary', 'Size']].to_frame().T
            prediction = log_reg_model.predict(input_df)[0]
            probability = log_reg_model.predict_proba(input_df)[0, 1]

            results.append({
                'CustomerId': row['CustomerId'],
                'calculated_rfms': {
                    'Recency': row['Recency'],
                    'Frequency': row['Frequency'],
                    'Monetary': row['Monetary'],
                    'Size': row['Size']
                },
                'prediction': int(prediction),  # 0 for bad, 1 for good
                'probability': float(probability)
            })
        
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)