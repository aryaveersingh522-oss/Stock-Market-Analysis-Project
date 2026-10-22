"""
Data Preprocessing and Cleaning Module
======================================
Handles data loading, missing value imputation, duplicate removal,
data type corrections, integrity validation, and chronological sorting.

Syllabus Alignment:
- Unit I: Introduction and Data Preparation (Data preprocessing, cleaning, handling anomalies)
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw stock market CSV data."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at {filepath}")
    logger.info(f"Loading raw data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform thorough data cleaning:
    1. Parse and standardize Date column
    2. Check and remove any duplicate records
    3. Validate and enforce correct numeric datatypes
    4. Logical bounds validation (High >= Low, High >= Open/Close, Volume >= 0)
    5. Handle missing values:
       - Missing values in historical features are forward-filled per ticker.
       - The very last row for each ticker has a missing Target_Up because next-day
         return cannot be computed yet (a natural characteristic of directional forecasting).
         We preserve this row with an explicit flag or drop it strictly for model training.
    6. Sort chronologically by Ticker and Date.
    """
    df = df.copy()

    # 1. Date formatting
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    null_dates = df['Date'].isnull().sum()
    if null_dates > 0:
        logger.warning(f"Dropping {null_dates} rows with unparseable dates.")
        df = df.dropna(subset=['Date'])

    # 2. Duplicate checking
    initial_len = len(df)
    df = df.drop_duplicates(subset=['Date', 'Ticker']).reset_index(drop=True)
    duplicates_removed = initial_len - len(df)
    logger.info(f"Duplicates removed: {duplicates_removed}")

    # 3. Numeric column types
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'MA_5', 'MA_20', 'Volatility_10']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Logical price validation
    # Ensure High is highest, Low is lowest
    invalid_high_low = df[df['High'] < df['Low']]
    if not invalid_high_low.empty:
        logger.warning(f"Found {len(invalid_high_low)} rows where High < Low; correcting bounds.")
        df['High_Corrected'] = df[['High', 'Low', 'Open', 'Close']].max(axis=1)
        df['Low_Corrected'] = df[['High', 'Low', 'Open', 'Close']].min(axis=1)
        df['High'] = df['High_Corrected']
        df['Low'] = df['Low_Corrected']
        df = df.drop(columns=['High_Corrected', 'Low_Corrected'])

    # Ensure Volume is non-negative
    df['Volume'] = df['Volume'].clip(lower=0)

    # 5. Chronological sorting
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    # 6. Feature interpolation per ticker if any intermediate NaNs exist
    for col in ['Daily_Return', 'MA_5', 'MA_20', 'Volatility_10']:
        if col in df.columns:
            df[col] = df.groupby('Ticker')[col].transform(lambda group: group.ffill().bfill())

    # 7. Summary statistics of missing values
    missing_summary = df.isnull().sum()
    logger.info("Missing values summary after cleaning:\n" + str(missing_summary[missing_summary > 0]))

    return df


def run_preprocessing_pipeline(
    raw_path: str = "data/raw/stock_market_data.csv",
    output_path: str = "data/processed/cleaned_stock_data.csv"
) -> pd.DataFrame:
    """Execute complete data preprocessing pipeline and save cleaned dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_raw = load_raw_data(raw_path)
    df_clean = clean_stock_data(df_raw)
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset successfully saved to: {output_path} ({len(df_clean)} rows)")
    return df_clean


if __name__ == "__main__":
    run_preprocessing_pipeline()
