import csv
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# --- Functions ---

def pred_rich_form(mol):
    poor_smi = Chem.MolToSmiles(mol, canonical=True)
    temp_mol = Chem.MolFromSmiles(poor_smi)
    Chem.Kekulize(temp_mol, clearAromaticFlags=True)
    for idx in range(temp_mol.GetNumBonds()):
        bond = temp_mol.GetBondWithIdx(idx)
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and bond.IsInRing():
            bond.SetBondType(Chem.rdchem.BondType.SINGLE)
    rich_smi = Chem.MolToSmiles(temp_mol, canonical=True, kekuleSmiles=False)
    rich_mol = Chem.MolFromSmiles(rich_smi)
    return rich_mol, rich_smi

def Cal_capH2(poor_smi):
    try:
        poor_mol = Chem.MolFromSmiles(poor_smi)
        if poor_mol is None:
            return None
        rich_mol, _ = pred_rich_form(poor_mol)
        MolWt_poor = Descriptors.MolWt(poor_mol)
        MolWt_rich = Descriptors.MolWt(rich_mol)
        capH2 = (MolWt_rich - MolWt_poor) / MolWt_rich * 100
        return round(capH2, 2)
    except Exception:
        return None

# --- Output file ---
output_file = "CapH2_selections.csv"

# --- Write header ---
with open(output_file, 'w', newline='') as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(['SMILES', 'CapH2 (%)'])

# --- Process multiple files ---
for i in range(1, 6):
    input_file = f"selection_canonical{i}.txt"
    print(f"📂 Processing {input_file}...")

    try:
        with open(input_file, 'r') as infile:
            smiles_list = [line.strip() for line in infile if line.strip()]

        with open(output_file, 'a', newline='') as out_csv:
            writer = csv.writer(out_csv)

            for smiles in tqdm(smiles_list, desc=f"File {i}"):
                capH2 = Cal_capH2(smiles)
                if capH2 is not None:
                    writer.writerow([smiles, capH2])

    except FileNotFoundError:
        print(f"⚠️ File not found: {input_file}. Skipping.")

print("✅ All files processed and saved to:", output_file)
