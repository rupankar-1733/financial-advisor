# config/settings.py
import os

# Project settings
PROJECT_NAME = "TCS Technical Analysis System"
VERSION = "1.0"

# Data settings
DEFAULT_DATA_PATH = os.path.join("data", "tcs_synthetic_5min.csv")
RESULTS_DIR = "results"

# Trading parameters
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
VOLUME_SPIKE_MULTIPLIER = 2.0
SIGNAL_STRENGTH_THRESHOLD = 2

# Schedule settings
MORNING_ANALYSIS_TIME = "09:30"
EVENING_ANALYSIS_TIME = "15:30"

# Email settings (update these)
EMAIL_ENABLED = False  # Set to True when ready
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"
ALERT_EMAIL = "your_email@gmail.com"
