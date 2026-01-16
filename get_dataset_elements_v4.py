
import re
import sys
import collections

# Simplified Regex for Element Extraction Only
# We don't need to tokenize bonds, numbers, or parens to find elements.
# We just need to find:
# 1. Bracketed atoms: [Na], [13C], [NH+]
# 2. Organic subset atoms: C, N, O, P, S, F, Cl, Br, I, B (and aromatic versions)

# Pattern for Bracketed items
P_BRACKETS = r"[^"]+"

# Pattern for Organic Subset (2-char first, then 1-char to respect greedy matching order if needed, but regex engine usually does strict left-to-right OR)
# We want to match 'Cl' before 'C'.
P_ORGANIC = r"Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"

# Combined Pattern
# We match brackets first, then organic atoms.
# Anything else (bonds, numbers, parens) will be ignored by findall.
PATTERN_STR = f"({P_BRACKETS}|{P_ORGANIC})"

SMILES_TOKENS_REGEX = re.compile(PATTERN_STR)
ELEMENT_PATTERN = re.compile(r"[A-Za-z]+")

def get_elements_from_smiles(smiles):
    elements_found = set()
    # findall returns a list of matches.
    # If groups are used, it returns tuples.
    # Our pattern has one outer capturing group `(...)`, so it returns strings.
    tokens = SMILES_TOKENS_REGEX.findall(smiles)
    
    for token in tokens:
        if token.startswith("["):
            content = token[1:-1]
            match = ELEMENT_PATTERN.search(content)
            if match:
                elem = match.group(0)
                clean_elem = elem.rstrip("H")
                if clean_elem:
                    elements_found.add(clean_elem.title())
        else:
            # It matches P_ORGANIC
            # Handle aromatic case (lowercase) -> Title case
            elements_found.add(token.title())
            
    return elements_found

def main():
    file_path = "data/2M.csv"
    print(f"Scanning {file_path} for elements...")
    
    unique_elements = collections.Counter()
    
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                if ',' in line:
                    smiles = line.split(',')[0].strip('"')
                else:
                    smiles = line.split()[0].strip('"')
                
                elems = get_elements_from_smiles(smiles)
                unique_elements.update(elems)
                
                if (i + 1) % 500000 == 0:
                    print(f"Processed {i + 1} lines...")
                    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    print("\n" + "="*40)
    print("ELEMENTS FOUND IN DATASET")
    print("="*40)
    
    for elem, count in unique_elements.most_common():
        print(f"{elem}: {count}")

if __name__ == "__main__":
    main()
