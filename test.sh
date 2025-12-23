#!/bin/bash
# Delete old submission, run model, and submit to Kaggle

set -e  # Exit on error

echo "🗑️  Deleting old submission.csv..."
rm -f submission.csv

echo "🚀 Running new.py..."
python new.py

echo "📤 Submitting to Kaggle..."
python submit_to_kaggle.py -m "LightGBM + Ridge blend, 180K recent data"

echo "✅ Done!"
