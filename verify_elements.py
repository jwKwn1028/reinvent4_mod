import sys
from reinvent.datapipeline.filters import elements

# 1. Simulate the default "Base Elements" defined in the code
print(f"Default Base Elements: {sorted(list(elements.BASE_ELEMENTS))}")

# 2. Simulate a user configuration that only adds "P" (Phosphorus)
user_config_elements = ["P"]
print(f"User Config Elements:  {user_config_elements}")

# 3. Replicate the EXACT logic from `reinvent/datapipeline/preprocess.py` (Line 90)
#    Code: config.filter.elements = list(elements.BASE_ELEMENTS | set(config.filter.elements))
combined_elements = list(elements.BASE_ELEMENTS | set(user_config_elements))

print(f"Combined Elements:     {sorted(combined_elements)}")

# 4. Verify that both a base element (Carbon) and the new element (Phosphorus) are accepted
test_elements = ["C", "P", "Au"] # Carbon (Base), Phosphorus (New), Gold (Unsupported)

print("\n--- Testing Filter Logic ---")
for elem in test_elements:
    is_supported = elem in combined_elements
    status = "ALLOWED" if is_supported else "REMOVED"
    print(f"Element '{elem}': {status}")
