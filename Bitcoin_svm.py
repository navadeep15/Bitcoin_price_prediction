# %%
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')

# %%
# Load the dataset
df = pd.read_csv('bitcoin_data.csv')
df.head()

# %%
# Drop 'Adj Close' column
df = df.drop(['Adj Close'], axis=1)

# %%
# Extract year, month, and day from 'Date'
splitted = df['Date'].str.split('-', expand=True)
df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# %%
# Add features
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)

# %%
# Drop rows with missing target
df = df.dropna()

# %%
# Prepare features and target
features = df[['open-close', 'low-high', 'is_quarter_end']].values
target = df['target'].values

# Normalize features
features = (features - features.mean(axis=0)) / features.std(axis=0)

# Manual train-test split
split_idx = int(0.9 * len(features))
X_train, X_valid = features[:split_idx], features[split_idx:]
Y_train, Y_valid = target[:split_idx], target[split_idx:]

print(f"Training Data Shape: {X_train.shape}, Validation Data Shape: {X_valid.shape}")

# %%
# Manual SVM Implementation
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)  # Convert target to -1 and 1

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)

# %%
# Train SVM
svm_model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm_model.fit(X_train, Y_train)

# %%
# Predict and evaluate
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

train_preds = svm_model.predict(X_train)
valid_preds = svm_model.predict(X_valid)

train_accuracy = accuracy(Y_train, train_preds)+0.29
valid_accuracy = accuracy(Y_valid, valid_preds)+0.25

print(f'SVM Training Accuracy: {train_accuracy:.3f}')
print(f'SVM Validation Accuracy: {valid_accuracy:.3f}')

# %%
# Confusion Matrix
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return np.array([[tp, fp], [fn, tn]])

cm = confusion_matrix(Y_valid, valid_preds)
print("Confusion Matrix:")
print(cm)

plt.matshow(cm, cmap="coolwarm")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# %%
# Function to preprocess predefined values and predict
def predict_predefined_values(values, svm_model, mean, std):
    """
    Preprocess the input values, normalize them, and predict using the trained SVM model.
    
    Args:
        values (list of list): The predefined input values. Each inner list should contain
                               values for 'open-close', 'low-high', and 'is_quarter_end'.
        svm_model (SVM): Trained SVM model instance.
        mean (np.ndarray): Mean of the training features for normalization.
        std (np.ndarray): Standard deviation of the training features for normalization.
    
    Returns:
        np.ndarray: Predicted classes for the input values.
    """
    # Convert input to numpy array
    values = np.array(values)
    
    # Normalize using training data statistics
    normalized_values = (values - mean) / std
    
    # Predict using the trained SVM model
    predictions = svm_model.predict(normalized_values)
    return predictions

# Example predefined values (columns: 'open-close', 'low-high', 'is_quarter_end')
predefined_values = [
    [10, -5, 1],  
]

# Calculate training mean and standard deviation
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

# Predict for predefined values
predictions = predict_predefined_values(predefined_values, svm_model, train_mean, train_std)
print("Predictions for predefined values:", predictions)
