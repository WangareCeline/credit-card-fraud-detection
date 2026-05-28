# Credit Card Fraud Detection
 
A machine learning project that builds a binary classification model to detect fraudulent financial transactions, exploring data preprocessing, class imbalance handling, model training and evaluation using Python and scikit-learn.
 
> I recently built this as part of my journey into machine learning, training a Random Forest classifier to identify fraudulent credit card transactions and experimenting with techniques like SMOTE to handle imbalanced data.
 
---

## Project Overview

Financial fraud detection is a classic machine learning problem characterized by highly imbalanced datasets, where fraudulent transactions represent a small fraction of total activity. This project investigates how well a Random Forest classifier can identify fraud, and compares three approaches to handling class imbalance.

---

## Dataset

**Source:** [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/code/abdelazizelserty/credit-card-fraud-detection-dataset?select=credit_card_fraud_10k.csv)

The dataset contains 10,000 financial transactions with the following features:

| Column | Description |
|--------|-------------|
| `transaction_id` | Unique transaction identifier (dropped because it has no predictive value) |
| `amount` | Transaction value |
| `transaction_hour` | Hour of day the transaction occurred |
| `merchant_category` | Category of merchant (Food, Clothing, Travel, Grocery, Electronics) |
| `foreign_transaction` | Whether the transaction occurred abroad (0/1) |
| `location_mismatch` | Whether location differs from cardholder's usual area (0/1) |
| `device_trust_score` | Trust score of the device used |
| `velocity_last_24h` | Number of transactions in the preceding 24 hours |
| `cardholder_age` | Age of the cardholder |
| `is_fraud` | **Target variable**: 1 = fraud, 0 = legitimate |

**Class distribution:**
- Legitimate transactions: 9,849 (98.5%)
- Fraudulent transactions: 151 (1.5%)

---

## Technologies Used

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Jupyter Notebook

---


## Methodology

### 1. Data Preprocessing
- Inspected dataset for missing values (none found)
- Applied Label Encoding to convert `merchant_category` from text to numeric
- Dropped `transaction_id` as it carries no predictive value
- Defined features (X) and target (y)

### 2. Train/Test Split
- 80% training (8,000 rows) / 20% testing (2,000 rows)
- `random_state=42` for reproducibility

### 3. Model Training
Trained a **Random Forest Classifier** with 100 decision trees as the baseline model.

### 4. Handling Class Imbalance
Three approaches were tested and compared:

| Approach | Description |
|----------|-------------|
| Baseline | Standard Random Forest with no imbalance handling |
| SMOTE | Synthetic Minority Oversampling (creates artificial fraud examples) |
| Class Weight | Penalizes the model more heavily for missing fraud cases |

---

## Results

### Model Comparison:

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| **Baseline** ✅ | 1.00 | **0.61** | **0.76** |
| SMOTE | 0.26 | 0.61 | 0.36 |
| Class Weight | 1.00 | 0.55 | 0.71 |

**The baseline Random Forest model performed best** (achieving perfect precision with the highest recall and F1 score among all three approaches.)

### Confusion Matrix (Baseline Model):

| | Predicted Legitimate | Predicted Fraud |
|--|---------------------|-----------------|
| **Actual Legitimate** | 1969 ✅ | 0 ✅ |
| **Actual Fraud** | 12 ❌ | 19 ✅ |

---

## Key Findings

- **Accuracy is misleading on imbalanced datasets.** A model predicting everything as legitimate achieves 98.5% accuracy while detecting zero fraud. Recall and F1 score are more meaningful metrics.
- **Recall is the priority in fraud detection.** Missing actual fraud is more costly than a false alarm.
- **Complexity does not guarantee improvement.** Both SMOTE and class weighting underperformed the baseline model on this dataset.
- **Always benchmark against a simple model** before introducing more complex solutions.

---


## Acknowledgements

Dataset provided by [Abdelaziz Elserty on Kaggle](https://www.kaggle.com/code/abdelazizelserty/credit-card-fraud-detection-dataset?select=credit_card_fraud_10k.csv).

---
