import pandas as pd
import argparse
import sys
import os
from rdkit import Chem
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Trim, rename CSV columns, and calculate NKetones.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    return parser.parse_args()

def calculate_nketones(smiles):
    ketone_pattern = Chem.MolFromSmarts('[CX3](=O)([#6])[#6]')
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            matches = mol.GetSubstructMatches(ketone_pattern)
            return len(matches)
        return 0
    except:
        return 0

def main():
    args = get_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    try:
        print(f"Reading {args.input}...")
        df = pd.read_csv(args.input)
        print(f"Original row count: {len(df)}")
        
        # 1. Filter Score if it exists
        if 'Score' in df.columns:
            print("Filtering rows where Score is 0...")
            # converting to numeric just in case, treating errors as NaN
            df['Score'] = pd.to_numeric(df['Score'], errors='coerce').fillna(0)
            df = df[df['Score'] != 0]
            print(f"Row count after removing Score=0: {len(df)}")
        else:
            print("'Score' column not found. Skipping Score filter (assuming already filtered).")

        # 2. Rename columns logic
        # Check if we need to rename
        raw_cols_map = {
            'DE (raw)': 'DE',
            'MP (raw)': 'MP',
            'BP (raw)': 'BP',
            'CapH2 (raw)': 'CapH2'
        }
        
        # Perform renaming only if the raw columns exist
        cols_to_rename = {k: v for k, v in raw_cols_map.items() if k in df.columns}
        if cols_to_rename:
            print(f"Renaming columns: {list(cols_to_rename.keys())}")
            df.rename(columns=cols_to_rename, inplace=True)
        else:
            print("No raw columns found. Checking if target columns already exist...")

        # 3. Calculate NKetone
        print("Calculating NKetone...")
        tqdm.pandas(desc="Counting Ketones")
        df['NKetone'] = df['SMILES'].progress_apply(calculate_nketones)

        # 4. Select final columns
        # We expect these columns to be present now
        target_columns = ['SMILES', 'DE', 'MP', 'BP', 'CapH2', 'NKetone']
        
        # Verify columns
        missing_cols = [col for col in target_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: The following columns are missing: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
            
        final_df = df[target_columns]
        
        print(f"Writing to {args.output}...")
        final_df.to_csv(args.output, index=False)
        print("Done.")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
