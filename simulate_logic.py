# ---------------------------------------------------------
# Simulation of REINVENT4 Logic
# ---------------------------------------------------------

# 1. From reinvent/datapipeline/filters/elements.py
BASE_ELEMENTS = {"C", "O", "N", "S", "F", "Cl", "Br", "I"}
print(f"Default Base Elements: {sorted(list(BASE_ELEMENTS))}")

# 2. Simulate your configuration in config.toml
#    [filter]
#    elements = ["P"]
user_config_elements = ["P"]
print(f"User Config Elements:  {user_config_elements}")

# 3. From reinvent/datapipeline/preprocess.py (Line 90)
#    The code performs a SET UNION (|) between base and config elements.
#    This guarantees that config elements are ADDED, not replacing.
combined_elements = list(BASE_ELEMENTS | set(user_config_elements))

print(f"Combined Elements:     {sorted(combined_elements)}")

# 4. Simulation of the Filter Check
print("\n--- Testing Filter Logic ---")
test_elements = ["C", "P", "Au"]

for elem in test_elements:
    # Logic from regex.py: if elem not in self.elements: return None
    is_supported = elem in combined_elements
    status = "ALLOWED" if is_supported else "REMOVED"
    print(f"Element '{elem}': {status}")
