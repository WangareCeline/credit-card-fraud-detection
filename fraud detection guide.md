# Fraud Detection ML Project - Full Guide

A complete walkthrough of building a binary classification model to detect fraudulent financial transactions using Python and scikit-learn. Covers data preprocessing, model training, evaluation, and class imbalance handling.

---

## 1. Dataset Overview

The dataset contains 10,000 financial transactions with the following columns:

| Column | Description |
|--------|-------------|
| `transaction_id` | Unique identifier for each transaction - not useful for ML, dropped later |
| `amount` | Transaction value - fraudsters often transact in unusual amounts |
| `transaction_hour` | Time of day - fraud often happens at odd hours |
| `merchant_category` | Type of shop or service - some categories attract more fraud |
| `foreign_transaction` | Whether the transaction happened abroad |
| `location_mismatch` | Whether the location differs from the cardholder's usual area |
| `device_trust_score` | How trusted the device used is - low score is suspicious |
| `velocity_last_24h` | Number of transactions in the last 24 hours - high velocity is a red flag |
| `cardholder_age` | Age of the cardholder |
| `is_fraud` | **Target column** - 1 means fraud, 0 means legitimate |

**Note:** Only 1.5% of transactions are fraud (151 out of 10,000). This is called a **class imbalance** and requires careful handling.

---

## 2. Setting Up & Loading Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('your_file_name_here.csv')
df.head()
```

**What this does:**
- Imports all necessary libraries for data handling and visualization
- Loads the dataset into a Pandas DataFrame
- `df.head()` shows the first 5 rows to confirm it loaded correctly

---

## 3. Exploring the Data

```python
df.shape
```
Returns **(10000, 10)** - 10,000 rows and 10 columns.

```python
df.info()
```
**What to look for:**
- **Non-Null Count** - checks for missing values. All 10,000 rows are complete with no missing data.
- **Dtype** - data types. `merchant_category` is `object` (text) and needs converting. Everything else is numeric.

```python
df['is_fraud'].value_counts()
```
Returns:
- 0 (legitimate): 9,849
- 1 (fraud): 151

This confirms the **class imbalance** - the model needs to handle this carefully or it will just predict "not fraud" every time.

---

## 4. Encoding Categorical Data

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['merchant_category'] = le.fit_transform(df['merchant_category'])
```

**Why we do this:**
ML models are mathematical equations and they cannot process text like "Food" or "Travel." Label Encoding converts text categories into numbers:

| Category | Encoded Value |
|----------|--------------|
| Clothing | 0 |
| Electronics | 1 |
| Food | 2 |
| Grocery | 3 |
| Travel | 4 |

Any text column must be converted before feeding it to a model.

---

## 5. Dropping Irrelevant Columns

```python
df = df.drop('transaction_id', axis=1)
```

**Why:** `transaction_id` is just a unique identifier with no predictive value. Including it would confuse the model without adding any useful information.

---

## 6. Defining Features and Target

```python
features = ["amount", "transaction_hour", "merchant_category", "foreign_transaction",
            "location_mismatch", "device_trust_score", "velocity_last_24h", "cardholder_age"]
X = df[features]
y = df.is_fraud
```

**What this means:**
- **X (features)** - the input columns the model learns from
- **y (target)** - what the model is trying to predict (`is_fraud`)

Explicitly listing features is safer than using `df.drop('is_fraud')`. If new columns are added to the dataset later, X will not accidentally include something unintended.

---

## 7. Splitting into Train and Test Sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**What this does:**
- **80% (8,000 rows)** - training data the model learns from
- **20% (2,000 rows)** - test data the model is evaluated on, data it has never seen before

Think of it like studying for an exam. The 8,000 rows are the study material and the 2,000 are the actual test.

**`random_state=42`** ensures the split is reproducible so anyone running the same code gets the same split.

---

## 8. Training the Model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

**What is Random Forest?**
A collection of 100 decision trees (`n_estimators=100`) that each vote on whether a transaction is fraud. The majority vote wins. It handles imbalanced data reasonably well and is widely used in fraud detection.

`model.fit(X_train, y_train)` is where the actual learning happens. The model studies the training data and finds patterns.

---

## 9. Evaluating the Model

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Results:

| Metric | Legitimate (0) | Fraud (1) |
|--------|---------------|-----------|
| Precision | 0.99 | 1.00 |
| Recall | 1.00 | 0.61 |
| F1 Score | 1.00 | 0.76 |
| Accuracy | | 0.99 |

### What each metric means:

**Precision** - Of all the transactions the model flagged as fraud, how many were actually fraud?
- Fraud precision = 1.00, meaning every time it said fraud, it was right. Zero false alarms.

**Recall** - Of all the actual fraud cases, how many did the model catch?
- Fraud recall = 0.61, meaning it caught 61% of real fraud and missed 39%.

**F1 Score** - The balance between precision and recall. Useful for imbalanced datasets.

**Accuracy** - Overall correct predictions. 99% sounds great but is misleading here because 98% of the data is already legitimate.

In fraud detection, recall matters most. Missing actual fraud costs the bank real money. A model that catches more fraud even with a few false alarms is more valuable than a precise but blind one.

---

## 10. Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### Reading the matrix:

| | Predicted Legitimate | Predicted Fraud |
|--|---------------------|-----------------|
| **Actual Legitimate** | 1969 | 0 |
| **Actual Fraud** | 12 | 19 |

- **1969** - Legitimate transactions correctly identified
- **0** - Legitimate transactions wrongly flagged as fraud (no false alarms)
- **12** - Fraud cases the model missed
- **19** - Fraud cases correctly caught

---

## 11. Handling Class Imbalance - Three Approaches Compared

### Approach 1: SMOTE (Synthetic Minority Oversampling Technique)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

**What it does:** Creates synthetic fraud examples to balance the dataset, resulting in 7,880 legitimate vs 7,880 fraud cases.

**Result:** Recall stayed at 61% but precision dropped from 100% to 26%. More false alarms without catching more fraud.

---

### Approach 2: Class Weight Balancing

```python
model_weighted = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)
```

**What it does:** Tells the model to penalize missed fraud cases more heavily during training without creating new data.

**Result:** Precision stayed at 100% but recall dropped to 55%, which is slightly worse than the original.

---

### Final Comparison - All Three Models:

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Original** | 1.00 | 0.61 | 0.76 |
| SMOTE | 0.26 | 0.61 | 0.36 |
| Class Weight | 1.00 | 0.55 | 0.71 |

The original model came out on top with the highest recall, perfect precision and the best F1 score.

More complexity does not always mean better results. Sometimes the simplest model wins.

---

## 12. Key ML Concepts Covered

| Concept | Definition |
|---------|------------|
| **Classification** | Predicting a category (fraud vs not fraud) |
| **Regression** | Predicting a number (use MAE, RMSE for this, not for classification) |
| **Imbalanced dataset** | When one class has far more examples than another |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Train/Test split** | Separating data for learning vs evaluation |
| **Label Encoding** | Converting text categories to numbers |
| **SMOTE** | Creating synthetic minority class examples to balance data |
| **Random Forest** | An ensemble of decision trees that vote on predictions |

---

## 13. Key Takeaways

**On model selection:**
When solving a classification problem, always compare multiple approaches before settling on a final model. Metrics like precision, recall and F1 score tell a more complete story than accuracy alone, especially on imbalanced datasets.

**On imbalanced data:**
A dataset where only 1.5% of cases are positive will produce misleading accuracy scores. A model predicting everything as negative achieves 98.5% accuracy while being completely useless. Always evaluate with recall and F1 score in these cases.

**On choosing the right metric:**
The right metric depends on the business problem. In fraud detection, missing actual fraud is more costly than a false alarm, so recall is prioritized. In other problems the trade-off may be different. Always ask what a wrong prediction actually costs.

**On complexity:**
More complex solutions do not always produce better results. Testing SMOTE and class weighting showed that the simplest baseline model outperformed both. Always benchmark against a simple model before adding complexity.

---

*Built as a learning project to explore classification, imbalanced datasets, and model evaluation in a real-world context.*
