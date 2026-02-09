import sys

from rdkit import Chem


def main():
    input_file = "2M.smi"
    output_file = "2M.smi"

    # Allowed atomic numbers: C(6), N(7), O(8), F(9), S(16), Cl(17), Br(35)+Si+P+Bss
    allowed_atomic_nums = {5, 6, 7, 8, 9, 14, 15, 16, 17, 35}

    print(f"Reading from {input_file}...")
    print(f"Writing to {output_file}...")

    count_total = 0
    count_kept = 0

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            original_line = line
            line = line.strip()
            if not line:
                continue

            # Assuming the first token is the SMILES string
            parts = line.split()
            smiles = parts[0]

            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                valid = True
                for atom in mol.GetAtoms():
                    if atom.GetAtomicNum() not in allowed_atomic_nums:
                        valid = False
                        break

                if valid:
                    f_out.write(original_line)
                    count_kept += 1

            count_total += 1
            if count_total % 10000 == 0:
                print(f"Processed {count_total} lines...", end="\r")

    print(f"\nDone. Processed {count_total} molecules. Kept {count_kept} molecules.")


if __name__ == "__main__":
    main()
