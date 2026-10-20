# Stock Market Analysis and Direction Prediction Project

## Project Overview
This repository contains an end-to-end Data Science and Machine Learning project analyzing Indian stock market data (`HDFCBANK.NS`, `ICICIBANK.NS`, `INFY.NS`, `RELIANCE.NS`, `TCS.NS`). The project encompasses:
- Data Preprocessing and Cleaning (handling missing values, data validation)
- Exploratory Data Analysis (investigating 5 key objectives with clean, simple visualizations: Line Charts, Bar Charts, Countplots)
- Feature Engineering (momentum, moving average ratios, lag returns)
- Supervised Machine Learning with 5 syllabus-aligned models:
  1. **Logistic Regression** (Linear baseline)
  2. **K-Nearest Neighbors (KNN)** (Lazy learning instance-based classifier)
  3. **Naïve Bayes (GaussianNB)** (Probabilistic classifier)
  4. **Decision Tree Classifier** (Divide-and-conquer rule-based tree model)
  5. **Support Vector Machine (SVM)** (Maximum margin classifier)
- Comprehensive Model Evaluation (Accuracy, Log Loss, ROC-AUC, Precision, Recall, F1 Score, Confusion Matrix)
- Final findings, strategic insights, and risk management recommendations.

## Directory Structure
```
Stock-Market-Analysis-Project/
├── data/
│   ├── raw/                  # Original raw stock dataset
│   └── processed/            # Cleaned & feature-engineered data
├── src/                      # Modular Python source code
│   ├── data_preprocessing.py # Cleaning and data validation
│   ├── eda.py                # Exploratory Data Analysis & visual charts
│   ├── feature_engineering.py# Technical feature extraction
│   ├── model_training.py     # Training 5 syllabus-aligned ML models
│   └── model_evaluation.py   # Performance evaluation & metric reporting
├── models/                   # Serialized model weights & scaler
├── notebooks/                # Jupyter notebook walkthrough
├── reports/
│   ├── figures/              # Simple, high-quality plots (line, bar, countplot)
│   └── final_findings_and_recommendations.md # Full project analytical report
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
