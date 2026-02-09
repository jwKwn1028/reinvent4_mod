import os
import sys
import tomli
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Ensure we can import from the reinvent package
sys.path.append(os.getcwd())

# -----------------------------------------------------------------------------
# Re-implementation of Transform Logic (to mimic REINVENT4)
# -----------------------------------------------------------------------------

def hard_sigmoid(x: np.ndarray, k: float) -> np.ndarray:
    return (k * x > 0).astype(np.float32)

def stable_sigmoid(x: np.ndarray, k: float, base_10: bool = True) -> np.ndarray:
    h = k * x
    if base_10:
        h = h * np.log(10)
    # Avoid overflow
    hp_idx = h >= 0
    y = np.zeros_like(x, dtype=float)
    y[hp_idx] = 1.0 / (1.0 + np.exp(-h[hp_idx]))
    y[~hp_idx] = np.exp(h[~hp_idx]) / (1.0 + np.exp(h[~hp_idx]))
    return y.astype(np.float32)

def double_sigmoid(x, x_left, x_right, k, k_left, k_right):
    x_center = (x_right - x_left) / 2 + x_left
    xl = x[x < x_center] - x_left
    xr = x[x >= x_center] - x_right

    if k == 0:
        sigmoid_left = hard_sigmoid(xl, k_left)
        sigmoid_right = 1 - hard_sigmoid(xr, k_right)
    else:
        k_left = k_left / k
        k_right = k_right / k
        sigmoid_left = stable_sigmoid(xl, k_left)
        sigmoid_right = 1 - stable_sigmoid(xr, k_right)

    d_sigmoid = np.zeros_like(x)
    d_sigmoid[x < x_center] = sigmoid_left
    d_sigmoid[x >= x_center] = sigmoid_right
    return d_sigmoid

class Transform:
    def __call__(self, values):
        pass

class Sigmoid(Transform):
    def __init__(self, low, high, k):
        self.low = low
        self.high = high
        self.k = k

    def __call__(self, values):
        values = np.array(values, dtype=np.float32)
        x = values - (self.high + self.low) / 2
        if (self.high - self.low) == 0:
            k_val = 10.0 * self.k
            return hard_sigmoid(x, k_val)
        else:
            k_val = 10.0 * self.k / (self.high - self.low)
            return stable_sigmoid(x, k_val)

class ReverseSigmoid(Transform):
    def __init__(self, low, high, k):
        self.low = low
        self.high = high
        self.k = k

    def __call__(self, values):
        values = np.array(values, dtype=np.float32)
        x = values - (self.high + self.low) / 2
        if (self.high - self.low) == 0:
            k_val = 10.0 * self.k
            return 1.0 - hard_sigmoid(x, k_val)
        else:
            k_val = 10.0 * self.k / (self.high - self.low)
            return 1.0 - stable_sigmoid(x, k_val)

class DoubleSigmoid(Transform):
    def __init__(self, low, high, coef_div=100.0, coef_si=150.0, coef_se=150.0):
        self.low = low
        self.high = high
        self.coef_div = coef_div
        self.coef_si = coef_si
        self.coef_se = coef_se

    def __call__(self, values):
        values = np.array(values, dtype=np.float32)
        return double_sigmoid(values, self.low, self.high, self.coef_div, self.coef_si, self.coef_se)

class ValueMapping(Transform):
    def __init__(self, mapping):
        self.mapping = {str(k): float(v) for k, v in mapping.items()}
        # For plotting, we need sorted keys if they are numeric
        try:
            self.numeric_keys = sorted([float(k) for k in self.mapping.keys()])
            self.is_numeric = True
        except ValueError:
            self.is_numeric = False
            self.numeric_keys = sorted(self.mapping.keys())

    def __call__(self, values):
        transformed = []
        for v in values:
            s = str(v)
            # Try to handle float strings like "1.0" matching "1" or vice versa if needed
            # For now exact match
            val = self.mapping.get(s, np.nan)
            if np.isnan(val) and self.is_numeric:
                # Try formatting as float then back to string to match keys
                 try:
                     s_float = str(float(v))
                     val = self.mapping.get(s_float, np.nan)
                 except:
                     pass
            transformed.append(val)
        return np.array(transformed)

# -----------------------------------------------------------------------------
# Main Visualizer
# -----------------------------------------------------------------------------

def load_config(path="configs/scores.toml"):
    with open(path, "rb") as f:
        return tomli.load(f)

def get_transform_from_config(t_config):
    t_type = t_config.get("type", "").lower()
    
    if t_type == "sigmoid":
        return Sigmoid(
            low=t_config.get("low", 0.0),
            high=t_config.get("high", 0.0),
            k=t_config.get("k", 0.0)
        )
    elif t_type == "reversesigmoid":
        return ReverseSigmoid(
            low=t_config.get("low", 0.0),
            high=t_config.get("high", 0.0),
            k=t_config.get("k", 0.0)
        )
    elif t_type == "doublesigmoid":
        return DoubleSigmoid(
            low=t_config.get("low", 0.0),
            high=t_config.get("high", 0.0),
            coef_div=t_config.get("coef_div", 100.0),
            coef_si=t_config.get("coef_si", 150.0),
            coef_se=t_config.get("coef_se", 150.0)
        )
    elif t_type == "valuemapping":
        return ValueMapping(
            mapping=t_config.get("mapping", {})
        )
    else:
        return None

def plot_component(name, transform_obj, output_path):
    plt.figure(figsize=(8, 6))
    
    if isinstance(transform_obj, ValueMapping):
        if transform_obj.is_numeric:
            keys = transform_obj.numeric_keys
            # Add some padding to x-axis
            x_min = min(keys) - 1
            x_max = max(keys) + 1
            x = np.linspace(x_min, x_max, 200)
            
            # For value mapping, we only have discrete points really, 
            # but let's try to map the linspace to show where matches occur?
            # Or better, just scatter plot the defined points.
            
            # Plot defined points
            x_points = keys
            y_points = [transform_obj.mapping[str(k)] for k in keys]
            # Handle keys stored as "1.0"
            y_points = []
            for k in keys:
                val = transform_obj.mapping.get(str(k))
                if val is None: val = transform_obj.mapping.get(str(float(k)))
                y_points.append(val)

            plt.scatter(x_points, y_points, color='red', zorder=5, label='Mapped Values')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel("Input Value")
            plt.ylabel("Score")
            plt.title(f"Component: {name} (ValueMapping)")
            
            # Optional: Show what happens between points (NaN or Default)?
            # REINVENT returns NaN/0 for missing keys usually.
        else:
            # Categorical
            keys = list(transform_obj.mapping.keys())
            values = list(transform_obj.mapping.values())
            plt.bar(keys, values)
            plt.xlabel("Category")
            plt.ylabel("Score")
            plt.title(f"Component: {name} (ValueMapping)")
            
    elif hasattr(transform_obj, 'low') and hasattr(transform_obj, 'high'):
        low = transform_obj.low
        high = transform_obj.high
        
        # Determine range
        span = abs(high - low)
        if span == 0: span = 1.0
        
        # Plot range: extend 50% beyond low/high
        x_min = min(low, high) - span * 0.5
        x_max = max(low, high) + span * 0.5
        
        if isinstance(transform_obj, DoubleSigmoid):
             # Double sigmoid usually needs a wider range to show the falloff
             x_min -= span * 0.5
             x_max += span * 0.5
             
        x = np.linspace(x_min, x_max, 500)
        y = transform_obj(x)
        
        plt.plot(x, y, linewidth=2)
        plt.axvline(x=low, color='r', linestyle=':', label=f'Low ({low})')
        plt.axvline(x=high, color='g', linestyle=':', label=f'High ({high})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xlabel("Input Value")
        plt.ylabel("Score")
        plt.title(f"Component: {name} ({type(transform_obj).__name__})")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def main():
    config = load_config()
    components = config.get("component", [])
    
    if not os.path.exists("vis"):
        os.makedirs("vis")

    for comp_entry in components:
        # Each entry is a dict like {'CompUnimol': {'endpoint': [...]}}
        # But 'endpoint' is a list of dicts.
        
        # Flatten structure
        for comp_type, data in comp_entry.items():
            endpoints = data.get("endpoint", [])
            for ep in endpoints:
                name = ep.get("name", "Unknown")
                t_config = ep.get("transform", {})
                
                if not t_config:
                    print(f"Skipping {name}: No transform defined.")
                    continue
                
                transform_obj = get_transform_from_config(t_config)
                if transform_obj:
                    # Sanitize filename
                    safe_name = "".join([c if c.isalnum() else "_" for c in name])
                    plot_component(name, transform_obj, f"vis/score_{safe_name}.png")
                else:
                    print(f"Skipping {name}: Unknown transform type {t_config.get('type')}")

if __name__ == "__main__":
    main()
