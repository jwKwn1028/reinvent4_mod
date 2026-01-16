
import polars as pl
import io

# Mock data: 3 lines of text
data = """C=CC(=O)Cc1ccccc1
OC[C@@H](O)CN1CCN(c2ccccc2)CC1
O=C(c1ccc(OC(F)F)cc1)N(OCC1CC1)c1c(Cl)cncc1Cl"""

# 1. Test with a standard separator (e.g., \t) which is NOT present
# This should read each line as a single row.
print("--- Test 1: Separator = '\t' (Default) ---")
try:
    df = pl.read_csv(io.BytesIO(data.encode()), separator='\t', has_header=False)
    print(df)
    print(f"Shape: {df.shape}")
except Exception as e:
    print(f"Error: {e}")

# 2. Test with Separator = '\n'
# This is unusual. Let's see if Polars allows it.
print("\n--- Test 2: Separator = '\n' ---")
try:
    df = pl.read_csv(io.BytesIO(data.encode()), separator='\n', has_header=False)
    print(df)
    print(f"Shape: {df.shape}")
except Exception as e:
    print(f"Error: {e}")
