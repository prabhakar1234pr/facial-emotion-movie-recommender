"""
Main Entry Point for Emotion Detection Application

Run this file to start the Streamlit application:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ui.streamlit_app import main

if __name__ == "__main__":
    main()

