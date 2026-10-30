"""
End-to-End Stock Market Analysis Pipeline Runner
================================================
Executes all stages sequentially:
1. Data Preprocessing & Cleaning (Unit I)
2. Exploratory Data Analysis & Visualizations (Unit I & II)
3. Feature Engineering (Unit I & II)
4. Model Training with TimeSeriesSplit CV (Unit II & III)
5. Model Evaluation & Best-Fit Model Selection (Unit III)
"""

import sys
import logging
from src.data_preprocessing import run_preprocessing_pipeline
from src.eda import run_eda
from src.feature_engineering import run_feature_engineering_pipeline
from src.model_training import run_training_pipeline
from src.model_evaluation import evaluate_all_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")


def main():
    logger.info("==================================================")
    logger.info("  STARTING END-TO-END DATA SCIENCE & ML PIPELINE  ")
    logger.info("==================================================")

    logger.info("\n--- STEP 1: DATA PREPROCESSING & CLEANING ---")
    run_preprocessing_pipeline()

    logger.info("\n--- STEP 2: EXPLORATORY DATA ANALYSIS (5 OBJECTIVES) ---")
    run_eda()

    logger.info("\n--- STEP 3: FEATURE ENGINEERING ---")
    run_feature_engineering_pipeline()

    logger.info("\n--- STEP 4: MODEL TRAINING (5 SYLLABUS MODELS) ---")
    run_training_pipeline()

    logger.info("\n--- STEP 5: MODEL EVALUATION & SELECTION ---")
    metrics_df = evaluate_all_models()

    logger.info("==================================================")
    logger.info("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!      ")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
