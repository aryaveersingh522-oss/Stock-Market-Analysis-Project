"""
Exploratory Data Analysis (EDA) Module
====================================
Investigates 5 core market objectives using standard, clean visualizations:
- Line charts
- Bar charts
- Countplots

Syllabus Alignment:
- Unit I & II: Descriptive statistics, trend identification, correlation concepts.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Set clean, accessible visualization style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def run_eda(
    input_path: str = "data/processed/cleaned_stock_data.csv",
    output_dir: str = "reports/figures"
):
    """Conduct comprehensive EDA and export simple, standard visual charts."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])

    logger.info("Executing EDA across 5 defined objectives...")

    # -------------------------------------------------------------
    # OBJECTIVE 1: Price Trends & Comparative Trajectory
    # Chart: Line Chart
    # -------------------------------------------------------------
    logger.info("Plotting Objective 1: Stock Price Trends (Line Chart)...")
    plt.figure(figsize=(10, 5))
    for ticker, group in df.groupby('Ticker'):
        plt.plot(group['Date'], group['Close'], label=ticker, linewidth=1.8)
    plt.title("Objective 1: Daily Stock Closing Prices (Jan - Oct 2021)", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Close Price (INR)", fontsize=11)
    plt.legend(title="Stock Ticker", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_stock_price_trends_linechart.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # OBJECTIVE 2: Moving Average Crossover (Trend Identification)
    # Chart: Line Chart (Close, MA_5, MA_20)
    # -------------------------------------------------------------
    logger.info("Plotting Objective 2: Moving Average Crossovers (Line Chart)...")
    hdfc_df = df[df['Ticker'] == 'HDFCBANK.NS']
    plt.figure(figsize=(10, 5))
    plt.plot(hdfc_df['Date'], hdfc_df['Close'], label="Daily Close", color='#1f77b4', linewidth=1.5, alpha=0.9)
    plt.plot(hdfc_df['Date'], hdfc_df['MA_5'], label="5-Day MA (Short Term)", color='#ff7f0e', linewidth=1.8, linestyle="--")
    plt.plot(hdfc_df['Date'], hdfc_df['MA_20'], label="20-Day MA (Medium Term)", color='#2ca02c', linewidth=2.0)
    plt.title("Objective 2: Moving Average Crossover for HDFCBANK.NS", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Price (INR)", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_moving_averages_linechart.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # OBJECTIVE 3: Target Direction Distribution (Market Up/Down Bias)
    # Chart: Countplot
    # -------------------------------------------------------------
    logger.info("Plotting Objective 3: Target Class Distribution (Countplot)...")
    valid_targets = df.dropna(subset=['Target_Up']).copy()
    valid_targets['Target_Label'] = valid_targets['Target_Up'].map({1.0: "Up Day (1)", 0.0: "Down Day (0)"})

    plt.figure(figsize=(9, 5))
    ax = sns.countplot(
        data=valid_targets,
        x="Ticker",
        hue="Target_Label",
        palette=["#e74c3c", "#2ecc71"]
    )
    plt.title("Objective 3: Next-Day Directional Distribution (Target_Up) by Stock", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Stock Ticker", fontsize=11)
    plt.ylabel("Number of Trading Days", fontsize=11)
    plt.legend(title="Direction", frameon=True)

    # Annotate bar heights
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height / 2),
                        ha='center', va='center', fontsize=9, color='white', weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_target_distribution_countplot.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # OBJECTIVE 4: Return and Volatility Risk Profile
    # Chart: Bar Charts
    # -------------------------------------------------------------
    logger.info("Plotting Objective 4: Average Daily Return & Volatility (Bar Charts)...")
    perf_df = df.groupby('Ticker').agg(
        Mean_Daily_Return=('Daily_Return', 'mean'),
        Return_Volatility=('Daily_Return', 'std')
    ).reset_index()

    # 4A: Average Daily Return Bar Chart
    plt.figure(figsize=(8, 4.8))
    colors = ['#27ae60' if r >= 0 else '#c0392b' for r in perf_df['Mean_Daily_Return']]
    bars = plt.bar(perf_df['Ticker'], perf_df['Mean_Daily_Return'] * 100, color=colors, edgecolor='black', alpha=0.85)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title("Objective 4A: Average Daily Return (%) by Stock Ticker", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Stock Ticker", fontsize=11)
    plt.ylabel("Average Daily Return (%)", fontsize=11)

    for bar in bars:
        h = bar.get_height()
        offset = 0.005 if h >= 0 else -0.015
        plt.text(bar.get_x() + bar.get_width()/2., h + offset, f"{h:.3f}%", ha='center', va='bottom', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_average_returns_barchart.png"), dpi=200)
    plt.close()

    # 4B: Return Volatility Bar Chart
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(perf_df['Ticker'], perf_df['Return_Volatility'] * 100, color='#3498db', edgecolor='black', alpha=0.85)
    plt.title("Objective 4B: Daily Return Volatility (Risk) by Stock Ticker", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Stock Ticker", fontsize=11)
    plt.ylabel("Standard Deviation of Returns (%)", fontsize=11)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f}%", ha='center', va='bottom', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_volatility_barchart.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # OBJECTIVE 5: Trading Volume Activity Across Stocks
    # Chart: Bar Chart
    # -------------------------------------------------------------
    logger.info("Plotting Objective 5: Trading Volume Activity (Bar Chart)...")
    vol_df = df.groupby('Ticker')['Volume'].mean().reset_index()
    vol_df['Volume_Millions'] = vol_df['Volume'] / 1e6

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(vol_df['Ticker'], vol_df['Volume_Millions'], color='#9b59b6', edgecolor='black', alpha=0.85)
    plt.title("Objective 5: Average Daily Trading Volume by Stock (Millions)", fontsize=13, weight='bold', pad=12)
    plt.xlabel("Stock Ticker", fontsize=11)
    plt.ylabel("Average Daily Volume (Millions of Shares)", fontsize=11)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f}M", ha='center', va='bottom', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_trading_volume_barchart.png"), dpi=200)
    plt.close()

    logger.info(f"EDA successfully completed. All charts exported to: {output_dir}")


if __name__ == "__main__":
    run_eda()
