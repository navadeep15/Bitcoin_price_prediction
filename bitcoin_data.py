import yfinance as yf
import pandas as pd

# Fetch Bitcoin data
btc_data = yf.download('BTC-USD', start='2010-01-01', end='2024-12-31')


# Fetch Bitcoin data with hourly interval  (IF YOU WANT TO FETCH THE DATA HOURLY:)
# btc_data = yf.download('BTC-USD', start='2010-01-01', end='2024-12-31', interval='1h')


# Display the first few rows of the data
print(btc_data.head())

# Save to a CSV file
btc_data.to_csv('bitcoin_data.csv')