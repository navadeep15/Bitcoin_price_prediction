import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('bitcoin_data.csv')

# Extract year, month, and day from 'Date'
splitted = df['Date'].str.split('-', expand=True)
df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

# Convert the 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Add quarter-end indicator
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

# Feature engineering
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = df['Close'].shift(-1)  # Predict the next close price
df['ma7'] = df['Close'].rolling(window=7).mean()
df['ma30'] = df['Close'].rolling(window=30).mean()
df['daily_return'] = df['Close'].pct_change()
df['volatility'] = df['Close'].rolling(window=7).std()

# Drop rows with NaN values caused by rolling calculations
df = df.dropna()

# Define features and target
features = df[['open-close', 'low-high', 'is_quarter_end', 'ma7', 'ma30', 'daily_return', 'volatility']].values
target = df['target'].values

# Split dataset into training and validation sets
split_index = int(len(features) * 0.9)
X_train, X_valid = features[:split_index], features[split_index:]
Y_train, Y_valid = target[:split_index], target[split_index:]

# Decision Tree Implementation
class DecisionTreeRegressorFromScratch:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'value': np.mean(y)}

        feature, threshold, score = self._find_best_split(X, y)
        if score is None:  # No valid split found
            return {'value': np.mean(y)}

        left_idx = X[:, feature] < threshold
        right_idx = ~left_idx

        return {
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_idx], y[left_idx], depth + 1),
            'right': self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }

    def _find_best_split(self, X, y):
        best_feature, best_threshold, best_score = None, None, float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] < threshold
                right_idx = ~left_idx

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                score = self._calculate_split_score(y[left_idx], y[right_idx])
                if score < best_score:
                    best_feature, best_threshold, best_score = feature, threshold, score

        return best_feature, best_threshold, best_score

    def _calculate_split_score(self, left, right):
        left_var = np.var(left) * len(left)
        right_var = np.var(right) * len(right)
        return left_var + right_var

    def _predict_one(self, x, tree):
        if 'value' in tree:
            return tree['value']
        if x[tree['feature']] < tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])

# Train the custom decision tree
dt_model = DecisionTreeRegressorFromScratch(max_depth=5, min_samples_split=10)
dt_model.fit(X_train, Y_train)

# Make predictions
train_pred = dt_model.predict(X_train)
valid_pred = dt_model.predict(X_valid)

# Evaluation functions
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Evaluate the model on training and validation sets
train_mae = np.mean(np.abs(Y_train - train_pred))
valid_mae = np.mean(np.abs(Y_valid - valid_pred))
train_rmse = root_mean_squared_error(Y_train, train_pred)
valid_rmse = root_mean_squared_error(Y_valid, valid_pred)
train_r2 = r_squared(Y_train, train_pred)
valid_r2 = r_squared(Y_valid, valid_pred)

print('Custom Decision Tree Regressor:')
print(f'Training MAE: {train_mae:.2f}')
print(f'Validation MAE: {valid_mae:.2f}')
print(f'Training RMSE: {train_rmse:.2f}')
print(f'Validation RMSE: {valid_rmse:.2f}')
print(f'Training R^2: {train_r2:.2f}')
print(f'Validation R^2: {valid_r2:.2f}')

# Visualize predictions
plt.figure(figsize=(10, 6))
plt.plot(range(len(Y_valid)), Y_valid, label='Actual Prices')
plt.plot(range(len(Y_valid)), valid_pred, label='Predicted Prices')
plt.legend()
plt.title('Actual vs Predicted Prices using Custom Decision Tree')
plt.show()

# Function to predict the next close price
def predict_next_close_custom(input_data):
    """
    Predict the next close price based on input features using the trained tree.

    Parameters:
    - input_data: Array-like, shape (n_features,)

    Returns:
    - Predicted close price
    """
    return dt_model.predict(np.array([input_data]))[0]

# Example usage
example_input = [100, 200, 1, 40000, 40500, 0.02, 500]  # Replace with actual input values
predicted_value = predict_next_close_custom(example_input)

print(f"Predicted next close price: {predicted_value:.2f}")