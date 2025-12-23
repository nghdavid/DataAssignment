#!/usr/bin/env python3
"""
Submit predictions to Kaggle competition.
Usage: python submit_to_kaggle.py [--message "Your message"] [--file submission.csv]

Setup:
1. Create a .env file in the same directory with:
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_api_key

2. Get your API key from: https://www.kaggle.com/settings/account
   (Scroll to "API" section -> "Create New Token")
"""

import subprocess
import sys
import argparse
import os
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Falling back to system environment variables or kaggle.json")


def setup_kaggle_credentials():
    """Set up Kaggle credentials from environment variables."""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if username and key:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        print(f"✅ Kaggle credentials loaded for user: {username}")
        return True
    else:
        print("⚠️  KAGGLE_USERNAME or KAGGLE_KEY not found in .env file")
        print("   Falling back to ~/.kaggle/kaggle.json")
        return False


def submit_to_kaggle(
    competition: str = "sc6117-crypto-forecast",
    submission_file: str = "submission.csv",
    message: str = None
) -> bool:
    """
    Submit a file to a Kaggle competition.

    Args:
        competition: Competition name/slug
        submission_file: Path to the submission CSV file
        message: Submission message (auto-generated if not provided)

    Returns:
        True if submission was successful, False otherwise
    """
    # Check if file exists
    if not Path(submission_file).exists():
        print(f"❌ Error: Submission file '{submission_file}' not found!")
        return False

    # Auto-generate message if not provided
    if message is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Submission at {timestamp} - LightGBM + Ridge blend"

    # Build the command
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", competition,
        "-f", submission_file,
        "-m", message
    ]

    print(f"📤 Submitting to Kaggle...")
    print(f"   Competition: {competition}")
    print(f"   File: {submission_file}")
    print(f"   Message: {message}")
    print()

    try:
        # Run the submission command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            print("✅ Submission successful!")

            # Try to get submission status
            print("\n📊 Checking submission status...")
            status_cmd = [
                "kaggle", "competitions", "submissions",
                "-c", competition
            ]
            status_result = subprocess.run(
                status_cmd,
                capture_output=True,
                text=True
            )
            if status_result.stdout:
                # Print only the first few lines (most recent submissions)
                lines = status_result.stdout.strip().split('\n')
                for line in lines[:5]:  # Header + 4 most recent
                    print(line)

            return True
        else:
            print(f"❌ Submission failed with return code {result.returncode}")
            return False

    except FileNotFoundError:
        print("❌ Error: Kaggle CLI not found!")
        print("   Install it with: pip install kaggle")
        print("   Then configure your API key: https://www.kaggle.com/docs/api")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Submit predictions to Kaggle competition"
    )
    parser.add_argument(
        "-f", "--file",
        default="submission.csv",
        help="Submission file (default: submission.csv)"
    )
    parser.add_argument(
        "-m", "--message",
        default=None,
        help="Submission message (auto-generated if not provided)"
    )
    parser.add_argument(
        "-c", "--competition",
        default="sc6117-crypto-forecast",
        help="Competition name (default: sc6117-crypto-forecast)"
    )

    args = parser.parse_args()

    # Setup credentials from .env
    setup_kaggle_credentials()

    success = submit_to_kaggle(
        competition=args.competition,
        submission_file=args.file,
        message=args.message
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
