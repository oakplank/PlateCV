#!/usr/bin/env python3
"""
Script to run the License Plate Detection Web Application

This script starts the Flask web server for testing the license plate detection model.
Make sure you have trained a model (best.pt should exist in runs/train/weights/) before running.

Usage:
    python run_web_app.py

The application will be available at: http://127.0.0.1:5000
"""

import sys
import os
from pathlib import Path

# Add the web_app directory to Python path
web_app_dir = Path(__file__).parent / 'web_app'
sys.path.insert(0, str(web_app_dir))

# Change to web_app directory so Flask can find templates and static files
os.chdir(web_app_dir)

# Import and run the Flask app
try:
    from app import app, load_model
    print("=" * 60)
    print("License Plate Detection Web Application")
    print("=" * 60)
    
    # Load model before starting server
    if load_model():
        print("Model loaded successfully!")
        print("Starting server...")
        print("Open your browser and go to: http://127.0.0.1:5000")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        app.run(debug=False, host='127.0.0.1', port=5000)
    else:
        print("Failed to load model. Please check if best.pt exists in runs/train/weights/")
        sys.exit(1)
    
except ImportError as e:
    print(f"Error importing Flask app: {e}")
    print("Make sure you have installed all dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting the application: {e}")
    sys.exit(1) 