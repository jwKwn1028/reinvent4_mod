
import re
import sys
import collections

# Safe construction of regex pattern to avoid tool corruption
# We want: r"[[^]]+]"
# \ = 92, [ = 91, ^ = 94, ] = 93
# Pattern: \ [ [ ^ ] ] + ]
P_BRACKETS = chr(92) + chr(91) + chr(91) + chr(94) + chr(93) + chr(93) + "+" + chr(93)

P_ORGANIC = r"Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"

PATTERN_STR = f"({P_BRACKETS}|{P_ORGANIC})"

SMILES_TOKENS_REGEX = re.compile(PATTERN_STR)
ELEMENT_PATTERN = re.compile(r"[A-Za-z]+")

def get_elements_from_smiles(smiles):
    elements_found = set()
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
