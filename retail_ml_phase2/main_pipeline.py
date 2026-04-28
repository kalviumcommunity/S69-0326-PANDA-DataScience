import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from src.utils import setup_logging, get_data_path, logger
from src.feature_engineering import build_features
from src.forecasting_model import train_forecast_model
from src.classification_model import (
    train_stockout_classifier,
    train_overstock_classifier,
    train_product_speed_classifier
)
from src.clustering_model import cluster_products

def run_pipeline():
    setup_logging()
    logger.info("Starting Full ML Pipeline...")
    
    # 1. Load Data
    data_path = get_data_path()
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        return
        
    raw_df = pd.read_csv(data_path)
    
    # 2. Build Features
    df = build_features(raw_df)
    
    # 3. Train Models
    logger.info("Training Forecasting Model...")
    train_forecast_model(df)
    
    logger.info("Training Classification Models...")
    train_stockout_classifier(df)
    train_overstock_classifier(df)
    train_product_speed_classifier(df)
    
    logger.info("Running Clustering Model...")
    cluster_products(df)
    
    logger.info("Pipeline Complete!")

if __name__ == "__main__":
    run_pipeline()
