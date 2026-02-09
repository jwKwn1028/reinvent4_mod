import pandas as pd
import argparse
import sys
import os
from rdkit import Chem
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Merge and filter CSV files with InChI deduplication.")
    parser.add_argument(
        "--inputs", 
        nargs="+", 
        required=True, 
        help="List of input CSV file paths."
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path for the merged output CSV file."
    )
    parser.add_argument(
        "--chunksize", 
        type=int, 
        default=100000, 
        help="Number of rows to process at a time."
    )
    return parser.parse_args()

def validate_headers(input_files, required_columns):
    """
    Validates that all input files have the same headers and contain the required columns.
    Returns the columns of the first file if valid.
    """
    first_file_columns = None
    
    print("Validating headers...")
    for file_path in tqdm(input_files, desc="Validating files"):
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
            
        try:
            # Read only the header
            df_iter = pd.read_csv(file_path, nrows=0)
            columns = list(df_iter.columns)
            
            # Check for required columns
            missing_columns = [col for col in required_columns if col not in columns]
            if missing_columns:
                print(f"Error: File '{file_path}' is missing required columns: {missing_columns}")
                sys.exit(1)

            if first_file_columns is None:
                first_file_columns = columns
            # We no longer strictly enforce that all files must have identical columns,
            # as long as the 'required_columns' are present.
            # This allows merging files where some have 'NKetones' and others don't.
                    
        except Exception as e:
            print(f"Error reading header of {file_path}: {e}")
            sys.exit(1)
            
    print("Headers validated successfully.")
    return first_file_columns

def get_inchi_key(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToInchiKey(mol)
        return None
    except:
        return None

def process_files(input_files, output_file, required_columns, chunksize):
    """
    Reads input files in chunks, filters them, accumulates in memory, 
    deduplicates by InChIKey (keeping highest Score), and writes to output.
    """
    # Initialize tqdm for pandas apply
    tqdm.pandas(desc="Calculating InChIKeys")
    
    print(f"Processing files with chunksize={chunksize}...")
    
    collected_chunks = []
    
    for file_path in input_files:
        print(f"Reading {file_path}...")
        try:
            with pd.read_csv(file_path, chunksize=chunksize) as reader:
                # Wrap reader with tqdm to show chunk progress
                for chunk in tqdm(reader, desc=f"Chunks in {os.path.basename(file_path)}"):
                    # Filter rows where LOHC (raw) == 1.0
                    if 'LOHC (raw)' in chunk.columns:
                        filtered_chunk = chunk[chunk['LOHC (raw)'] == 1.0].copy()
                    else:
                        filtered_chunk = pd.DataFrame() 

                    if filtered_chunk.empty:
                        continue
                        
                    # Keep only specified columns
                    filtered_chunk = filtered_chunk[required_columns]
                    
                    # Compute InChIKey for deduplication
                    # Using progress_apply to show progress for this computationally expensive step
                    filtered_chunk['temp_inchi_key'] = filtered_chunk['SMILES'].progress_apply(get_inchi_key)
                    
                    # Drop rows where InChI generation failed
                    filtered_chunk = filtered_chunk.dropna(subset=['temp_inchi_key'])
                    
                    collected_chunks.append(filtered_chunk)
                    
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            sys.exit(1)

    if not collected_chunks:
        print("No valid data found matching criteria.")
        return

    print("Concatenating data...")
    full_df = pd.concat(collected_chunks, ignore_index=True)
    
    print(f"Total rows before deduplication: {len(full_df)}")
    
    # Sort by Score descending so that 'first' is the highest score
    print("Sorting by Score...")
    full_df.sort_values(by='Score', ascending=False, inplace=True)
    
    # Deduplicate based on InChIKey
    print("Deduplicating by InChIKey...")
    full_df.drop_duplicates(subset='temp_inchi_key', keep='first', inplace=True)
    
    # Remove the temporary InChI column
    full_df.drop(columns=['temp_inchi_key'], inplace=True)
    
    print(f"Total rows after deduplication: {len(full_df)}")
    
    print(f"Writing to {output_file}...")
    full_df.to_csv(output_file, index=False)
    print("Done.")

def main():
    args = get_args()
    
    required_columns = [
        'Score', 
        'SMILES', 
        'DE (raw)', 
        'MP (raw)', 
        'BP (raw)', 
        'CapH2 (raw)', 
        'LOHC (raw)'
    ]
    
    # 1. Validate headers first
    validate_headers(args.inputs, required_columns)
    
    # 2. Process and merge
    process_files(args.inputs, args.output, required_columns, args.chunksize)

if __name__ == "__main__":
    main()
