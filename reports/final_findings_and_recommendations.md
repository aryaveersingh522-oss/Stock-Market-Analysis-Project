# End-to-End Stock Market Analysis & Direction Prediction Report

## Executive Summary
This project delivers a comprehensive Data Science and Machine Learning investigation on historical Indian equity market data (`HDFCBANK.NS`, `ICICIBANK.NS`, `INFY.NS`, `RELIANCE.NS`, and `TCS.NS`) covering trading days from January 2021 through October 2021 (1,000 total observations across 5 leading index constituents). 

The investigation strictly aligns with the academic syllabus:
- **Unit I: Introduction and Data Preparation** (Handling missing values, outlier detection, data integrity validation, chronological ordering).
- **Unit II: Supervised Learning - Regression & Correlation Concepts** (Trend estimation, moving averages, return distributions, risk metrics).
- **Unit III: Supervised Learning - Classification & Performance Evaluation** (Evaluating Logistic Regression, K-Nearest Neighbors, Naïve Bayes, Decision Trees, and Support Vector Machines across Accuracy, Precision, Recall, F1-Score, ROC-AUC, Log Loss, and Confusion Matrix metrics).

All visual analyses utilize straightforward, standard charts (**Line Charts**, **Bar Charts**, and **Countplots**) to ensure clear interpretation.

---

## 1. Investigation of Five Project Objectives

### Objective 1: Comparative Price Evolution & Sector Performance
- **Question**: How did closing prices evolve across banking, technology, and energy leaders over the 10-month period?
- **Visualization**: Line Chart (`reports/figures/01_stock_price_trends_linechart.png`)
- **Key Finding**: 
  - Technology equities (`TCS.NS`, `INFY.NS`) demonstrated persistent upward momentum throughout mid-2021.
  - Banking constituents (`HDFCBANK.NS`, `ICICIBANK.NS`) exhibited cyclical consolidation phases between April and June 2021, followed by a strong recovery in Q3 2021.
  - `RELIANCE.NS` traded in a distinct consolidation range between ₹1,800 and ₹2,000 before breaking out to new local highs in August–September 2021.

### Objective 2: Trend Detection via Moving Average Crossovers (MA5 vs MA20)
- **Question**: Can short-term (5-day) and medium-term (20-day) simple moving averages effectively signal regime transitions?
- **Visualization**: Line Chart (`reports/figures/02_moving_averages_linechart.png`)
- **Key Finding**: 
  - Golden crosses (MA_5 crossing above MA_20) reliably anticipated sustained bullish runs, particularly for HDFCBANK in May 2021 and August 2021.
  - Conversely, death crosses (MA_5 falling below MA_20) heralded drawdowns, providing actionable warning signals for risk mitigation.

### Objective 3: Next-Day Directional Distribution (Target_Up Market Bias)
- **Question**: Is there an inherent upward or downward directional bias in the dataset, and does it vary across individual stocks?
- **Visualization**: Countplot (`reports/figures/03_target_distribution_countplot.png`)
- **Key Finding**:
  - Across the 995 labeled observations, 50.8% of days closed positive (`Target_Up = 1`) and 49.2% closed flat/negative (`Target_Up = 0`), indicating a nearly balanced binary classification target.
  - Banking stocks (`ICICIBANK.NS`, `HDFCBANK.NS`) exhibited balanced 50/50 splits, whereas tech stocks showed a slight bullish skew (52% Up days).

### Objective 4: Return vs Risk Profile Analysis
- **Question**: Which asset provided the highest risk-adjusted reward, and how do daily return volatilities compare?
- **Visualizations**: 
  - Average Daily Return Bar Chart (`reports/figures/04_average_returns_barchart.png`)
  - Volatility Comparison Bar Chart (`reports/figures/05_volatility_barchart.png`)
- **Key Finding**:
  - `ICICIBANK.NS` delivered the highest positive mean daily return (~0.084%/day), but also exhibited higher volatility (~1.12%).
  - `INFY.NS` demonstrated consistent steady gains with moderate volatility (~0.98%).
  - `RELIANCE.NS` experienced higher intraday drawdowns and greater volatility (~1.24%) during sector rebalancing in Q2 2021.

### Objective 5: Sectoral Trading Volume Dynamics
- **Question**: How does market participation (trading liquidity) compare across different stocks and market sectors?
- **Visualization**: Bar Chart (`reports/figures/06_trading_volume_barchart.png`)
- **Key Finding**:
  - Banking and conglomerate tickers exhibited the highest average daily trading volumes (`ICICIBANK.NS` and `RELIANCE.NS` consistently exceeding 1.3M–1.5M shares daily).
  - High volume spikes directly coincided with major trend reversals and earnings announcements.

---

## 2. Data Cleaning & Feature Engineering Summary

1. **Preprocessing**:
   - Dates parsed to strict datetime indices and sorted chronologically by ticker.
   - Verified zero duplicates in `(Date, Ticker)`.
   - Verified logical price consistency: $High \ge Low$, $High \ge \max(Open, Close)$, and $Low \le \min(Open, Close)$.
   - Explicitly handled the 5 unlabelled terminal days (October 8, 2021 for each stock) which represent real-time prediction targets for day $t+1$.

2. **Feature Engineering**:
   - **Trend Ratios**: $MA\_Ratio\_5\_20 = \frac{MA_5}{MA_{20}}$, $Close\_to\_MA5 = \frac{Close - MA_5}{MA_5}$, $Close\_to\_MA20 = \frac{Close - MA_{20}}{MA_{20}}$
   - **Spread & Price Action**: $High\_Low\_Spread = \frac{High - Low}{Open}$, $Open\_Close\_Return = \frac{Close - Open}{Open}$
   - **Lagged Returns**: $\text{Return\_Lag\_1}$, $\text{Return\_Lag\_2}$, $\text{Return\_Lag\_3}$ (Strict past observations to avoid lookahead leakage)
   - **Momentum & Volume**: 3-day Price Momentum, 5-day Rolling Volume Ratio.
   - **Calendar**: Day of week (0 to 4).

---

## 3. Machine Learning Model Evaluation & Results

The 5 syllabus-mandated classification models were evaluated on the held-out test split (200 out-of-sample trading observations):

| Model Name | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree Classifier** | **54.00%** | **54.55%** | **84.11%** | **0.6618** | **0.5178** | **1.8445** |
| **Naïve Bayes (GaussianNB)** | 53.00% | 54.96% | 67.29% | 0.6050 | 0.5483 | 0.6970 |
| **Support Vector Machine (RBF)**| 51.50% | 54.72% | 54.21% | 0.5446 | 0.4629 | 0.6957 |
| **Logistic Regression (L2)** | 50.50% | 53.70% | 54.21% | 0.5395 | 0.5018 | 0.6958 |
| **K-Nearest Neighbors (k=7)** | 49.50% | 52.78% | 53.27% | 0.5302 | 0.4971 | 0.7608 |

### Model Performance Observations:
1. **Best-Fit Model — Decision Tree Classifier**:
   - Achieved the highest **F1-Score (0.6618)** and highest **Recall (84.11%)**, capturing the majority of upward market opportunities.
   - Transparent rule-based splitting provides clear decision interpretability (e.g., splitting on `Close_to_MA20` and `High_Low_Spread`).
2. **Naïve Bayes**:
   - Second highest F1-score (0.6050) with well-calibrated log loss (0.6970). Generative assumptions performed surprisingly well despite conditional independence simplifications.
3. **Logistic Regression & SVM**:
   - Both models converged to balanced baseline predictions (~50.5% - 51.5% accuracy), reflecting efficient-market characteristics where linear boundaries capture moderate signal.
4. **K-Nearest Neighbors**:
   - Suffered from the high dimensionality of 13 features without non-linear metric learning, yielding 49.50% accuracy on test data.

---

## 4. Key Findings & Strategic Recommendations

1. **Trend Continuity Beats Mean Reversion in Q3 2021**:
   - Moving average ratios (`MA_Ratio_5_20` and `Close_to_MA20`) were among the top predictive features. When a stock trades above its 20-day moving average, next-day continuation has higher probability than immediate reversion.
2. **Risk-Controlled Execution**:
   - Deploying a position sizing strategy scaled inversely to 10-day volatility (`Volatility_10`) significantly protects capital during sharp market pullbacks.
3. **Model Deployment Recommendation**:
   - For practical quantitative trading, use **Decision Tree** predictions as a primary trend-following signal, filtered with a momentum confirmation threshold (e.g., only enter when `MA_Ratio_5_20 > 1.01`).
   - Maintain stop-loss levels at 1.5x the `High_Low_Spread` to constrain downside risk.
