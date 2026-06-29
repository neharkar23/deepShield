import os
import random
import logging
import sqlite3
import numpy as np
import torch
from datetime import datetime
from src.config import Config

# Initialize Logger
def get_logger(name="DeepShield"):
    """Configures and returns a logger that outputs to console and a file."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # File Handler
        log_file = os.path.join(Config.REPORTS_DIR, "deepshield.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

logger = get_logger()

# Set Global Seed for Reproducibility
def set_seed(seed: int = Config.SEED):
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global random seed set to {seed}")

# Database Helper Functions
def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(Config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the prediction history database table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

def log_prediction(image_name: str, prediction: str, confidence: float):
    """Logs a single model prediction to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO predictions (image_name, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (image_name, prediction, confidence, timestamp))
    conn.commit()
    conn.close()
    logger.info(f"Logged prediction: {image_name} -> {prediction} ({confidence*100:.2f}%)")

def get_prediction_history():
    """Retrieves all rows from the prediction history database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def clear_prediction_history():
    """Clears all records in the predictions table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    logger.info("Cleared all records from prediction history.")
