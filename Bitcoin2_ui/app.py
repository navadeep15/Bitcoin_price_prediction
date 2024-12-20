from flask import Flask, render_template, request
import joblib
import numpy as np
import random

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Reasons for Bitcoin price movements
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
    "Sell-offs triggered by profit-taking.",
    "Negative market sentiment due to unfavorable news.",
    "Low trading volumes indicating weak interest.",
    "High volatility causing uncertainty among traders.",
    "Increased regulatory scrutiny dampening sentiment.",
    "Wider economic instability impacting risk assets.",
    "Rising competition from alternative cryptocurrencies.",
    "Technological issues affecting network performance.",
    "Mining costs increasing without price compensation.",
    "Geopolitical tensions creating market fears."
]

# Function to predict the next close price
def predict_next_close(model, input_data):
    input_data_scaled = scaler.transform([input_data] if len(input_data.shape) == 1 else input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    reason = None
    if request.method == 'POST':
        try:
            # Get the input values from the form
            open_close = float(request.form['open_close'])
            low_high = float(request.form['low_high'])
            is_quarter_end = int(request.form['is_quarter_end'])
            ma7 = float(request.form['ma7'])
            ma30 = float(request.form['ma30'])
            daily_return = float(request.form['daily_return'])
            volatility = float(request.form['volatility'])

            # Create the input feature array
            input_data = np.array([open_close, low_high, is_quarter_end, ma7, ma30, daily_return, volatility])

            # Make the prediction
            prediction = predict_next_close(model, input_data)

            # Choose a reason based on the predicted price direction
            reason = random.choice(reasons_increase if prediction > 0 else reasons_decrease)
        except Exception as e:
            reason = f"An error occurred: {str(e)}"

    return render_template('index.html', prediction=prediction, reason=reason)

if __name__ == '__main__':
    app.run(debug=True)
