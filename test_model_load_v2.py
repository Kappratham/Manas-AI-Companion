#!/usr/bin/env python3
"""Test script to verify that the Manas model can be loaded with the updated HF_MODEL_REPO."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check what HF_MODEL_REPO is set to
hf_model_repo = os.getenv("HF_MODEL_REPO")
print(f"HF_MODEL_REPO is set to: {hf_model_repo}")

if not hf_model_repo:
    print("ERROR: HF_MODEL_REPO is not set!")
    sys.exit(1)

# Try to import and create a Manas instance
print("Attempting to import Manas class...")
try:
    from eval.conversation_runner import create_manas
    print("Successfully imported Manas class")
except Exception as e:
    print(f"ERROR importing Manas class: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create a Manas instance with small parameters for testing
print("Attempting to create Manas instance...")
try:
    manas = create_manas(max_new_tokens=10, temperature=0.7)
    print("SUCCESS: Successfully created Manas instance!")
    print(f"Manas model loaded from: {hf_model_repo}")
except Exception as e:
    print(f"ERROR creating Manas instance: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)