"""
Feature Engineering Module
==========================
Engineers technical momentum, moving average ratios, lag returns,
and intraday spread indicators without future lookahead leakage.

Syllabus Alignment:
- Unit I & II: Feature transformation, domain-specific feature derivation.
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer comprehensive technical features grouped by ticker:
    1. Trend & Moving Average Ratios (MA5, MA20)
    2. Lagged Returns (Lag 1, 2, 3)
    3. Intraday Spreads & Momentum
    4. Volume Indicators
    5. Calendar features
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    engineered_list = []

    for ticker, group in df.groupby('Ticker'):
        g = group.copy()

        # 1. Moving average ratios
        g['MA_Ratio_5_20'] = g['MA_5'] / (g['MA_20'] + 1e-8)
        g['Close_to_MA5'] = (g['Close'] - g['MA_5']) / (g['MA_5'] + 1e-8)
        g['Close_to_MA20'] = (g['Close'] - g['MA_20']) / (g['MA_20'] + 1e-8)

        # 2. Intraday price action spreads
        g['High_Low_Spread'] = (g['High'] - g['Low']) / (g['Open'] + 1e-8)
        g['Open_Close_Return'] = (g['Close'] - g['Open']) / (g['Open'] + 1e-8)

        # 3. Lagged returns (Strictly prior days to prevent lookahead)
        g['Return_Lag_1'] = g['Daily_Return'].shift(1)
        g['Return_Lag_2'] = g['Daily_Return'].shift(2)
        g['Return_Lag_3'] = g['Daily_Return'].shift(3)

        # 4. Multi-day Price Momentum
        g['Price_Momentum_3D'] = (g['Close'] - g['Close'].shift(3)) / (g['Close'].shift(3) + 1e-8)

        # 5. Volume dynamics
        g['Volume_MA5'] = g['Volume'].rolling(window=5, min_periods=1).mean()
        g['Volume_Ratio'] = g['Volume'] / (g['Volume_MA5'] + 1e-8)

        # 6. Calendar feature
        g['Day_Of_Week'] = g['Date'].dt.dayofweek

        # Backward fill initial lag NaNs to keep data usable
        for col in ['Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Price_Momentum_3D']:
            g[col] = g[col].bfill()

        engineered_list.append(g)

    result_df = pd.concat(engineered_list, ignore_index=True)
    logger.info(f"Engineered {len(result_df.columns)} total columns across {len(result_df)} rows.")
    return result_df


def run_feature_engineering_pipeline(
    input_path: str = "data/processed/cleaned_stock_data.csv",
    output_path: str = "data/processed/features_engineered_stock_data.csv"
) -> pd.DataFrame:
    """Execute complete feature engineering pipeline and export processed CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(input_path)
    engineered_df = engineer_features(df)
    engineered_df.to_csv(output_path, index=False)
    logger.info(f"Engineered dataset successfully saved to: {output_path}")
    return engineered_df


if __name__ == "__main__":
    run_feature_engineering_pipeline()
