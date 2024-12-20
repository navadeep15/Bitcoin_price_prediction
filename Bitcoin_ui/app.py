from flask import Flask, render_template, request
import pickle
import numpy as np
import random

# Load the trained model and scaler
model = pickle.load(open('bitcoin_model.pkl', 'rb'))
scaler = pickle.load(open('b_scaler.pkl', 'rb'))

app = Flask(__name__)

# Reasons for stock price going up or down
reasons_increase = [
    "Increased trading volume indicates strong market confidence.",
    "Positive market sentiment due to favorable news.",
    "High demand from institutional investors.",
    "Economic indicators suggest strong growth.",
    "Improved adoption rates of Bitcoin for transactions.",
    "Upcoming halving event boosting scarcity perception.",
    "Lower mining costs increasing profitability for miners.",
    "Regulatory clarity supporting broader acceptance.",
    "Positive earnings or quarterly reports related to crypto companies.",
    "Technological advancements improving transaction efficiency."
]

reasons_decrease = [
    "Reduced trading volume showing weak market confidence.",
    "Negative news impacting market sentiment.",
    "Large sell-off by institutional investors.",
    "Economic indicators suggest downturn or uncertainty.",
    "Hacking incidents or security concerns in the crypto space.",
    "Regulatory crackdowns reducing adoption rates.",
    "Increased competition from alternative cryptocurrencies.",
    "High mining costs causing reduced miner profitability.",
    "Geopolitical tensions causing market uncertainty.",
    "Technological delays or failures reducing confidence."
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input values from the form
        open_close = float(request.form['open_close'])
        low_high = float(request.form['low_high'])
        is_quarter_end = int(request.form['is_quarter_end'])
        
        # Prepare input for the model
        input_data = np.array([open_close, low_high, is_quarter_end]).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        
        # Make predictions
        predicted_class = model.predict(input_data_scaled)[0]
        predicted_proba = model.predict_proba(input_data_scaled)[0]
        
        # Select a random reason based on the predicted class
        reason = random.choice(reasons_increase if predicted_class == 1 else reasons_decrease)
        
        # Prepare response
        result = {
            'predicted_class': 'Increase' if predicted_class == 1 else 'Decrease',
            'probability_increase': round(predicted_proba[1] * 100, 2),
            'probability_decrease': round(predicted_proba[0] * 100, 2),
            'reason': reason
        }
        
        return render_template('result.html', result=result)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
