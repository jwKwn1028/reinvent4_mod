#!/bin/env python3
#
# This is an example how to use the ExternalProcess scoring component using
# RASCore.  The scripts expects a list of SMILES from stdin and will
# write a JSON string to stdout.

import sys
import json
import os

# Redirect stdout to stderr immediately to prevent ANY import logs or 
# initialization logs from corrupting the JSON output.
# This is critical for TensorFlow/Keras.
original_stdout = sys.stdout
sys.stdout = sys.stderr

try:
    # Import here (while stdout is redirected)
    from RAscore import RAscore_NN
    
    # Path to your model
    RASCORE_MODEL = "/home/intern_01/jwkwon/rascore/RAscore/RAscore/models/models/DNN_chembl_fcfp_counts/model.h5"

    # Initialize scorer
    # If this fails, it prints to stderr and exits non-zero (caught by try/except)
    nn_scorer = RAscore_NN.RAScorerNN(RASCORE_MODEL)

except Exception as e:
    print(f"Error initializing RAscore: {e}", file=sys.stderr)
    sys.exit(1)

finally:
    # We keep stdout redirected until we are ready to print JSON
    pass

# Read SMILES from stdin
try:
    # Read all lines first to know count
    lines = sys.stdin.readlines()
    smilies = [line.strip() for line in lines]
except Exception as e:
    print(f"Error reading stdin: {e}", file=sys.stderr)
    sys.exit(1)

scores = []

# Score loop
for i, smiles in enumerate(smilies):
    try:
        if not smiles:
            # Empty line? Return NaN or 0
            scores.append(0.0) 
            continue

        score = nn_scorer.predict(smiles)  # returns numpy.float32
        scores.append(float(score))
        
    except Exception as e:
        print(f"Error predicting for SMILES #{i} ({smiles}): {e}", file=sys.stderr)
        # CRITICAL: Append NaN so the list length matches input length
        scores.append(float("nan"))

# Restore stdout for JSON output
sys.stdout = original_stdout

# Format the JSON string for REINVENT4 and write it to stdout
data = {"version": 1, "payload": {"RAScore": scores}}

print(json.dumps(data))