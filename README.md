# Diabetes Risk Prediction using Machine Learning

A Machine Learning project for predicting the risk of diabetes using the **CDC BRFSS 2015 Health Indicators Dataset**. The project explores the relationships between health indicators and diabetes, compares multiple tree-based machine learning models, and deploys the final model in an interactive Streamlit dashboard.

---

## Overview

Early detection of diabetes is essential for preventing severe health complications. This project develops an interpretable machine learning model that predicts whether an individual is at risk of diabetes based on demographic, lifestyle, and health-related factors.

Unlike many healthcare prediction projects that prioritize accuracy alone, this work focuses on **high recall**, reducing the chance of missing patients who actually have diabetes.

---

## Features

- Comprehensive Exploratory Data Analysis (EDA)
- Correlation analysis for different variable types
- Multicollinearity analysis using VIF
- Comparison of multiple tree-based machine learning models
- Hyperparameter tuning with RandomizedSearchCV
- Feature importance analysis
- Decision Tree interpretation
- Interactive Streamlit dashboard
- Diabetes risk prediction application

---

# Dataset

## Source

CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015

The dataset contains health survey information collected from adults in the United States.

### Dataset Statistics

- **253,680 observations**
- **21 predictor variables**
- Binary classification target:
  - Diabetes
  - No Diabetes

Target variable:

```
Diabetes_binary
```

---

# Features

The dataset contains three types of variables.

## Continuous

- BMI
- MentHlth
- PhysHlth

## Binary

- HighBP
- HighChol
- CholCheck
- Smoker
- Stroke
- HeartDiseaseorAttack
- PhysActivity
- Fruits
- Veggies
- HvyAlcoholConsump
- AnyHealthcare
- NoDocbcCost
- DiffWalk
- Sex

## Ordinal

- GenHlth
- Education
- Income
- AgeGroup

---

# Exploratory Data Analysis

The project performs extensive EDA including:

- Distribution analysis
- Class imbalance analysis
- Continuous variable visualization
- Binary variable visualization
- Ordinal variable visualization
- Diabetes prevalence by feature
- Top diabetes risk factors

---

# Correlation Analysis

Different statistical methods were selected based on variable types.

| Variable Types | Method |
|---------------|--------|
| Binary vs Binary | Phi Correlation |
| Binary vs Continuous | Point-Biserial |
| Binary + Ordinal | Spearman |
| Continuous | Spearman |

Additional analysis:

- Heatmaps
- Target correlation
- Multicollinearity (VIF)

---

# Machine Learning Models

The following models were evaluated:

- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost

---

# Model Training

Pipeline:

1. Train / Validation / Test split
2. K-Fold Cross Validation
3. RandomizedSearchCV
4. Hyperparameter tuning
5. Class Weighting
6. Final evaluation on the test set

No over-sampling or under-sampling was used in order to preserve the real-world class distribution.

---

# Evaluation Metrics

Since the dataset is imbalanced, Accuracy was not used as the primary metric.

Models were evaluated using:

- Recall
- Precision
- F1-score
- ROC Curve
- Precision-Recall Curve

The project prioritizes **Recall**, as missing a diabetic patient (False Negative) has much greater consequences than incorrectly flagging a healthy individual.

---

# Results

## Best Model

**Decision Tree**

Reasons for selection:

- High Recall
- Minimal overfitting
- Easy to interpret
- Suitable for healthcare applications

Although ensemble methods achieved competitive performance, the Decision Tree was selected because transparency and interpretability are critical in medical decision-making.

---

# Feature Importance

The Decision Tree identified the following variables as the most influential predictors:

| Rank | Feature |
|------|---------|
| 1 | GenHlth |
| 2 | HighBP |
| 3 | BMI |
| 4 | AgeGroup |
| 5 | HighChol |

These findings are consistent with established clinical knowledge regarding diabetes risk factors.

---

# Dashboard

A Streamlit dashboard was developed to visualize the dataset and demonstrate the trained model.

### Dashboard Pages

- Overview
- Ordinal vs Diabetes
- Binary vs Diabetes
- Continuous vs Diabetes
- Correlation Analysis
- Top Risk Factors
- Diabetes Risk Prediction

---

# Diabetes Prediction

Users can input health information including:

- BMI
- Age
- Blood Pressure
- Cholesterol
- Physical Activity
- Smoking
- Fruit Consumption
- Vegetable Consumption
- General Health
- Income
- Education
- and other BRFSS variables

The application predicts:

- Diabetes Risk
- Prediction Probability

---

# Tech Stack

### Language

- Python

### Libraries

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- Streamlit

---

# Project Structure

```
.
├── data/
│   ├── diabetes_binary.csv
│
├── notebooks/
│
├── preprocessing/
│
├── models/
│
├── dashboard/
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

# Running the Project

## Clone repository

```bash
git clone https://github.com/yourusername/Diabetes-Risk-Prediction.git

cd Diabetes-Risk-Prediction
```

## Install requirements

```bash
pip install -r requirements.txt
```

## Launch dashboard

```bash
streamlit run app.py
```

---

# Future Improvements

- SHAP explainability
- Calibration analysis
- Probability threshold optimization
- LightGBM & CatBoost comparison
- Cost-sensitive learning
- External dataset validation
- Deep learning comparison
- Cloud deployment

---

# Authors

- **Đỗ Thái Gia Hy**
- Trần Huỳnh Huy Thông
- Hà Quang Đại
- Thái Thủy Đức

University of Economics Ho Chi Minh City (UEH)

---

# License

This project was developed for educational and research purposes.
