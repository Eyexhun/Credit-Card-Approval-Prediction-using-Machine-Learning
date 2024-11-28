# Credit Card Approval Prediction

This project is aimed at predicting credit card approval using various machine learning models. The dataset used contains information about credit card applicants, and the objective is to predict whether an applicant will be approved for a credit card or not.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Data Description](#data-description)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Hyperparameter Tuning with GridSearchCV](#Hyperparameter-Tuning-with-GridSearchCV)
- [Metrics](#metrics)
- [Usage](#usage)

## Overview

In this project, we use different machine learning algorithms to predict credit card approval. We apply **Logistic Regression**, **XGBoost**, **LightGBM**, and **Random Forest** models to the dataset. We also fine-tune these models using **GridSearchCV** for hyperparameter optimization and evaluate the performance of each model using several metrics, including **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**.

## Requirements

To run the code, you will need to install the following Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `seaborn`

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm seaborn
```
## Data Description

The dataset used in this project is `crxdata.csv`, which contains applicant information and whether the credit card application was approved (`+`) or not approved (`-`). The target variable is stored in column `A16`, while the other columns contain applicant features.

### Dataset Structure
- **Features**: Multiple columns representing applicant information.
- **Target**: Column `A16` with values `+` (approved) and `-` (not approved).

## Preprocessing

### 1. Missing Data Handling
- Any `?` values in the dataset are replaced with `NaN`.
- **Categorical columns** are imputed with the mode value (most frequent value).
- **Numerical columns** are imputed with the mean value.

### 2. Encoding
- The target variable (`A16`) is encoded to `1` (approved) and `0` (not approved).
- Categorical features are one-hot encoded.

### 3. Feature Scaling
- All numerical features are standardized using `StandardScaler` to ensure they are on the same scale.

## Model Training

Four machine learning models are used to predict credit card approval:

- **Logistic Regression**
- **XGBoost**
- **LightGBM**
- **Random Forest**

These models are trained on the preprocessed dataset.

## Hyperparameter Tuning with GridSearchCV

**GridSearchCV** is used to search for the best hyperparameters for each model using cross-validation. The following hyperparameter grids are used:

### Logistic Regression:
- `C`: Regularization strength, values: `[0.1, 1, 10]`
- `solver`: Optimization algorithm, values: `['liblinear', 'lbfgs']`
- `max_iter`: Maximum iterations, values: `[100, 200]`

### XGBoost:
- `learning_rate`: Learning rate, values: `[0.01, 0.1, 0.2]`
- `n_estimators`: Number of boosting rounds, values: `[100, 200, 500]`
- `max_depth`: Maximum depth of trees, values: `[3, 6, 10]`
- `subsample`: Subsample ratio of training instances, values: `[0.8, 1.0]`
- `colsample_bytree`: Subsample ratio of columns, values: `[0.8, 1.0]`

### LightGBM:
- `learning_rate`: Learning rate, values: `[0.01, 0.05, 0.1]`
- `n_estimators`: Number of boosting rounds, values: `[100, 200]`
- `max_depth`: Maximum depth of trees, values: `[5, 10, -1]`
- `num_leaves`: Number of leaves in a tree, values: `[31, 50, 100]`
- `subsample`: Subsample ratio, values: `[0.8, 1.0]`

### Random Forest:
- `n_estimators`: Number of trees, values: `[100, 200, 500]`
- `max_depth`: Maximum depth of trees, values: `[5, 10, 20]`
- `min_samples_split`: Minimum number of samples required to split a node, values: `[2, 5, 10]`
- `min_samples_leaf`: Minimum number of samples required to be at a leaf node, values: `[1, 2, 4]`
- `bootstrap`: Whether bootstrap samples are used, values: `[True, False]`

## Metrics

The models are evaluated using the following metrics:

- **Accuracy**: The proportion of correctly predicted instances.
- **Precision**: The proportion of true positives among all positive predictions.
- **Recall**: The proportion of true positives among all actual positives.
- **F1-Score**: The harmonic mean of precision and recall.
- **ROC-AUC**: The area under the ROC curve, representing the trade-off between true positive rate and false positive rate.

After training, the models are evaluated on the test set, and the best performing models are selected based on these metrics.

## Usage

### 1. Prepare the Dataset
Ensure that the dataset `crxdata.csv` is in the same directory as the script. The dataset does not have column headers, as they are added programmatically.

### 2. Run the Script
Once the required libraries are installed and the dataset is prepared, you can run the script. The script will:
- Load and preprocess the data.
- Train the models and perform grid search for hyperparameter tuning.
- Evaluate and print the results for each model.

### 3. Output
The output will be a comparison of the models' evaluation metrics, such as accuracy, precision, recall, F1-score, and ROC-AUC.

| Model               | Accuracy | Precision | Recall   | F1-Score | ROC-AUC |
|---------------------|----------|-----------|----------|----------|---------|
| Logistic Regression | 0.7681   | 0.7672    | 0.8091   | 0.7876   | 0.7654  |
| XGBoost             | 0.8551   | 0.8704    | 0.8545   | 0.8624   | 0.8551  |
| LightGBM            | 0.8551   | 0.8636    | 0.8636   | 0.8636   | 0.8545  |
| Random Forest       | 0.8647   | 0.8475    | 0.9091   | 0.8772   | 0.8618  |

## Conclusion

Based on the evaluation metrics of the four models, we can summarize the following insights:

1. **Random Forest** outperformed all other models with the highest **Accuracy** (0.8647), **Recall** (0.9091), and **F1-Score** (0.8772). It also demonstrated a strong **ROC-AUC** (0.8618), indicating good model performance across different thresholds. This model is particularly good at minimizing false negatives, which is crucial for tasks like credit card approval prediction.

2. **XGBoost** showed competitive results, achieving a **high Accuracy** (0.8551), **Precision** (0.8704), and **F1-Score** (0.8624). With a **ROC-AUC** of 0.8551, it is a reliable model, especially for scenarios where a balance between precision and recall is important.

3. **LightGBM** also demonstrated similar performance to XGBoost, with an **Accuracy** of 0.8551, **Precision** (0.8636), and **Recall** (0.8636). While its **ROC-AUC** (0.8545) was slightly lower than XGBoost, it still performed well, especially in terms of recall, making it suitable for problems requiring high sensitivity.

4. **Logistic Regression** performed adequately with an **Accuracy** of 0.7681 and a **Recall** of 0.8091, but it lagged behind the other models in terms of **Precision**, **F1-Score**, and **ROC-AUC**. It may be useful as a baseline model or for scenarios where a simpler, less computationally intensive approach is needed.

Overall, **Random Forest** appears to be the best-performing model for this credit card approval prediction task, followed closely by **XGBoost** and **LightGBM**. Depending on the specific requirements of the application (e.g., precision vs. recall trade-off), either of these models can be chosen for deployment.
