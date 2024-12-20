### Bitcoin Price Prediction Project

This project involves predicting Bitcoin prices using two distinct approaches: classification and regression. Below, you'll find a detailed description of how to set up and use the project, including obtaining the dataset, preprocessing techniques, and user-friendly interfaces for making predictions.

---

#### Step 1: Obtain the Dataset

To generate the dataset required for the project, run the following Python script:

```bash
python bitcoin_data.py
```

This will create a file named `bitcoin_data.csv`, which contains the Bitcoin price data.

---

#### Step 2: Approaches Used in Prediction

1. **Classification**:
   - This approach predicts whether the Bitcoin price will increase or decrease.
   - Output: A binary prediction (e.g., "Price will go up" or "Price will go down").
   - Models used: Logistic Regression and Support Vector Machine (SVM).

2. **Regression**:
   - This approach predicts the exact value of the Bitcoin price.
   - Output: A numerical prediction representing the Bitcoin price.
   - Models used: Linear Regression and Decision Tree.

**Note:** All models, including Linear Regression, Logistic Regression, SVM, and Decision Tree, were implemented without using the `sklearn` library. This involved custom coding for training and evaluation.

---

#### Step 3: Data Preprocessing

We implemented and utilized various preprocessing techniques to clean and prepare the data for the machine learning models. These techniques include:

- **Feature Engineering**: Created new features such as `open-close` (difference between open and close prices), `low-high` (difference between low and high prices), and a binary `is_quarter_end` feature to indicate quarter-end months.
- **Handling Missing Values**: Verified and confirmed there were no missing values in the dataset.
- **Column Selection**: Dropped unnecessary columns like `Adj Close` to simplify the dataset.
- **Date Feature Extraction**: Extracted year, month, and day components from the `Date` column for better temporal analysis.
- **Feature Scaling**: Applied manual scaling to standardize the features for model training.
- **Target Variable Creation**: Defined a binary target variable indicating whether the closing price increased the next day.

These preprocessing steps ensured the data was clean, relevant, and ready for the machine learning models.

---

#### Step 4: User Interfaces

This section utilizes the Flask framework to develop the user interfaces for both classification and regression predictions.

We developed user-friendly interfaces to allow users to make predictions easily. Each approach has its own dedicated interface.

##### Classification UI

To use the interface for classification:

1. Navigate to the `Bitcoin_ui` directory:
   ```bash
   cd Bitcoin_ui
   ```

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and go to:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

4. Use the UI to input data and predict whether the Bitcoin price will go up or down.

5. To stop the application, return to the terminal and press `Ctrl + C`.

6. Navigate back to the main directory:
   ```bash
   cd ..
   ```

##### Regression UI

To use the interface for regression:

1. Navigate to the `Bitcoin2_ui` directory:
   ```bash
   cd Bitcoin2_ui
   ```

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and go to:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

4. Use the UI to input data and predict the exact Bitcoin price.

5. To stop the application, return to the terminal and press `Ctrl + C`.

6. Navigate back to the main directory:
   ```bash
   cd ..
   ```
---

### Results:

To see the results see the `bitcoin_price_prediction.pdf`

This project combines machine learning techniques with practical user interfaces, making it accessible for both technical and non-technical users to explore Bitcoin price predictions.

