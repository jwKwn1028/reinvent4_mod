import pandas as pd
import os
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors

# --- USER CONFIGURATION -----------------------------------------------------
# REPLACE 'your_data.csv' with the actual name of your CSV file containing SMILES
INPUT_CSV = 'data/mp/MP.csv' 

# Name of the column inside that CSV that contains the SMILES strings
SMILES_COL = 'smiles' 

# The file you uploaded containing the list of 200 descriptors
DESCRIPTOR_INDEX = 'eda/rd_desc_normalized_index.csv'

# Output file name
OUTPUT_CSV = 'data/mp/MP_features.csv'
# ----------------------------------------------------------------------------

def main():
    print("--- RDKit Feature Calculator Starting ---")

    # 1. Verify Input Files Exist
    if not os.path.exists(INPUT_CSV):
        print(f"\n[ERROR] Input file not found: '{INPUT_CSV}'")
        print("Please open the script and change 'INPUT_CSV' to match your filename.")
        return

    if not os.path.exists(DESCRIPTOR_INDEX):
        print(f"\n[ERROR] Descriptor index file not found: '{DESCRIPTOR_INDEX}'")
        print("Make sure 'rd_desc_normalized_index.csv' is in the same folder.")
        return

    # 2. Load the list of feature names
    print(f"Loading descriptor names from {DESCRIPTOR_INDEX}...")
    try:
        desc_df = pd.read_csv(DESCRIPTOR_INDEX)
        # Extract the list of names (assuming the column is named 'name')
        target_features = desc_df['name'].tolist()
        print(f"-> Found {len(target_features)} descriptors to calculate.")
    except Exception as e:
        print(f"[ERROR] Could not read descriptor file: {e}")
        return

    # 3. Load the SMILES data
    print(f"Loading SMILES data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
        if SMILES_COL not in df.columns:
            print(f"\n[ERROR] Column '{SMILES_COL}' not found in {INPUT_CSV}.")
            print(f"-> Columns found: {list(df.columns)}")
            print("Please update the 'SMILES_COL' variable in the script.")
            return
        print(f"-> Loaded {len(df)} molecules.")
    except Exception as e:
        print(f"[ERROR] Could not read input CSV: {e}")
        return

    # 4. Prepare Calculation Function
    # We pre-fetch the functions to make the loop faster
    calc_funcs = []
    for name in target_features:
        if hasattr(Descriptors, name):
            calc_funcs.append((name, getattr(Descriptors, name)))
        else:
            print(f"   [Warning] Descriptor '{name}' not found in RDKit, skipping.")

    # 5. Run Calculation
    print("\nStarting calculation... (This might take a while for large files)")
    
    # We use a list of dicts for efficiency, then convert to DataFrame
    results = []
    
    for i, smile in enumerate(df[SMILES_COL]):
        mol = Chem.MolFromSmiles(str(smile))
        row_data = {}
        
        if mol:
            for name, func in calc_funcs:
                try:
                    row_data[name] = func(mol)
                except Exception:
                    row_data[name] = None
        else:
            # Handle invalid SMILES
            for name, _ in calc_funcs:
                row_data[name] = None

        results.append(row_data)

        # Progress indicator every 100 rows
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} / {len(df)} rows...", end='\r')

    print(f"Processed {len(df)} / {len(df)} rows. Done.          ")

    # 6. Merge and Save
    print("Merging results...")
    features_df = pd.DataFrame(results)
    final_df = pd.concat([df, features_df], axis=1)

    print(f"Saving to {OUTPUT_CSV}...")
    final_df.to_csv(OUTPUT_CSV, index=False)
    print("\n[SUCCESS] Process completed successfully.")

if __name__ == "__main__":
    main()