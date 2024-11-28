# Credit Approval Classification with Logistic Regression

This project involves analyzing and classifying credit approvals using Logistic Regression. The process includes data preprocessing, exploratory data analysis (EDA), and hyperparameter tuning with GridSearchCV.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Overview
The project aims to:
1. Analyze and preprocess credit approval data.
2. Train a Logistic Regression model to classify credit applications.
3. Tune hyperparameters to optimize model performance.

---

## Dataset
The dataset used is `crxdata.csv`, which contains anonymized features for credit approval. Key preprocessing steps include:
- Assigning column names.
- Handling missing values (`?` replaced with `NaN`).
- Encoding categorical variables.
- Standardizing numerical features.

### Target Variable
- **`A16`**: Binary classification of credit approval (1 for approved, 0 for not approved).

---

## Features
- **Exploratory Data Analysis (EDA)**:
  - Visualize target distribution.
  - Identify and handle missing values.
  - Analyze data types and correlations.
- **Preprocessing**:
  - Impute missing values.
  - Encode categorical variables with one-hot encoding.
  - Standardize numerical data.
- **Model Building**:
  - Logistic Regression with default parameters.
  - Hyperparameter tuning using GridSearchCV.
- **Evaluation**:
  - Confusion matrix and accuracy score.

---

## Installation
To run the project, ensure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `seaborn`
- `scikit-learn`

Install them using:
```bash
pip install pandas numpy seaborn scikit-learn
