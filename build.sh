#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies (Tesseract)
apt-get update
apt-get install -y tesseract-ocr

# Run your Python build
pip install -r requirements.txt