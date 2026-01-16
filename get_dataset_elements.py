import re
import sys
import collections
import pathlib

# Regex patterns adapted from REINVENT4 source code (reinvent/datapipeline/filters/regex.py)
# This ensures we identify tokens exactly as REINVENT does.

BRACKETS = r"[^\]+]"
ALIPHATIC = r"Br?|Cl?|N|O|S|P|F|I"
AROMATIC = r"b|c|n|o|s|p"
BONDS = r"-|=|#|\$|:|" # Removed escaped backslash and forward slash
BRANCH = r"\(|"
LABELS = r"%\d{2}|\d"
MISC = r"\.|*"

SMILES_TOKENS_REGEX = re.compile(
    rf"({BRACKETS}|{ALIPHATIC}|{AROMATIC}|{BONDS}|{BRANCH}|{LABELS}|{MISC})"
)

# Regex to extract element symbol from within a bracket token (e.g., [13CH3] -> C)
# Matches: Start of string or non-alpha chars, followed by Element Symbol, followed by end or non-lower case
# This is a simplified approximation. A better one is to strip non-alpha and look up known elements.
# REINVENT logic: 
# if match := ELEMENT.search(new_token):
#     elem = match.group(0).rstrip("H").title()

ELEMENT_PATTERN = re.compile(r"[A-Za-z]+")

# Standard Periodic Table (Subset for validation, but we want to discover ALL)
# We will just capture any alpha string that looks like an element.

def get_elements_from_smiles(smiles):
    """
    Parses a SMILES string and returns a set of elements found.
    """
    elements_found = set()
    tokens = SMILES_TOKENS_REGEX.findall(smiles)
    
    for token in tokens:
        # 1. Handle Bracketed Atoms: [Na+], [13C], [NH4+]
        if token.startswith("["):
            # Remove brackets
            content = token[1:-1]
            # Simple heuristic: The first alphabetic sequence is usually the element
            # (e.g. 'Na' in 'Na+', 'C' in '13C', 'N' in 'NH4+')
            match = ELEMENT_PATTERN.search(content)
            if match:
                elem = match.group(0)
                # Handle cases like 'NH' -> N
                if len(elem) > 1 and elem[1].isupper():
                     # Case like 'NH' inside bracket usually means N and H attached
                     # But standard elements are 1 or 2 chars, 2nd is lower.
                     # If we see 'NH', it's likely Nitrogen.
                     # Let's use a simpler heuristic: Title case it and strip common suffixes if needed?
                     # No, let's just take the first Capital + optional lowercase
                     pass
                
                # REINVENT Logic:
                # elem = match.group(0).rstrip("H").title()
                # This handles 'NH' -> 'N', 'C' -> 'C', 'Na' -> 'Na'
                clean_elem = elem.rstrip("H")
                if clean_elem: # Handle case where it might be just 'H' inside brackets?
                    elements_found.add(clean_elem.title())
        
        # 2. Handle Organic Subset (Aliphatic/Aromatic)
        elif len(token) > 0 and token[0].isalpha():
            if token.lower() in ['b', 'c', 'n', 'o', 's', 'p', 'f', 'i', 'cl', 'br']:
                 elements_found.add(token.title())
            else:
                # Catch-all for other unbracketed atoms if any (rare in SMILES)
                elements_found.add(token.title())
                
    return elements_found

def main():
    file_path = "data/2M.csv"
    
    print(f"Scanning {file_path} for elements...")
    
    unique_elements = collections.Counter()
    
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                smiles = line.strip().split()[0] # Assume first column is SMILES
                if not smiles: continue
                
                elems = get_elements_from_smiles(smiles)
                unique_elements.update(elems)
                
                if (i + 1) % 100000 == 0:
                    print(f"Processed {i + 1} lines...")
                    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    print("\n" + "="*40)
    print("ELEMENTS FOUND IN DATASET")
    print("="*40)
    
    # Sort by frequency
    for elem, count in unique_elements.most_common():
        print(f"{elem}: {count}")

if __name__ == "__main__":
    main()
