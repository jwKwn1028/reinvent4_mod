"""
Exploratroy Data Analysis with pandas, scikit learn and rdkit, on csv
"""
import pandas as pd
import json
import sys
import os
import numpy as np

try:
    from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

def analyze_csv_pandas(input_file, output_file, target_column=None, top_n=5, corr_threshold=0.9):
    try:
        # Load the CSV
        # dtype=str prevents pandas from automatically converting "001" to 1
        df = pd.read_csv(input_file, dtype=str)
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols
        missing_cells = int(df.isna().sum().sum())
        
        results = {
            "headers": df.columns.tolist(),
            "metadata": {
                "total_rows": total_rows,
                "columns": total_cols,
                "total_cells": total_cells,
                "missing_cells": missing_cells,
                "missing_percent": round((missing_cells / total_cells * 100) if total_cells else 0.0, 2),
                "duplicate_rows": int(df.duplicated().sum()),
                "memory_bytes": int(df.memory_usage(deep=True).sum())
            },
            "column_non_numerics": {},
            "column_non_numeric_count": {},
            "column_numeric_count": {},
            "column_stats": {},
            "column_missing": {},
            "column_unique": {},
            "column_mode": {},
            "column_top_values": {},
            "column_string_length": {},
            "column_entropy": {},
            "correlations": {},
            "sklearn": {},
            "rdkit": {},
            "first_column_duplicates": []
        }

        # --- 1. Find Redundancies in the First Column ---
        first_col = df.iloc[:, 0]
        # Find values that appear more than once
        duplicates = first_col[first_col.duplicated()].unique()
        results["first_column_duplicates"] = duplicates.tolist()

        # --- 2. Process Each Column ---
        numeric_columns = []
        numeric_data = {}

        for col in df.columns:
            col_series = df[col]

            # Missing and unique stats
            non_null = col_series.dropna()
            missing_count = int(total_rows - len(non_null))
            results["column_missing"][col] = {
                "missing": missing_count,
                "missing_percent": round((missing_count / total_rows * 100) if total_rows else 0.0, 2)
            }
            unique_count = int(non_null.nunique(dropna=True))
            results["column_unique"][col] = {
                "unique": unique_count,
                "unique_percent": round((unique_count / total_rows * 100) if total_rows else 0.0, 2)
            }

            # Mode and top values
            if not non_null.empty:
                value_counts = non_null.value_counts(dropna=True)
                mode_value = value_counts.index[0]
                mode_count = int(value_counts.iloc[0])
                results["column_mode"][col] = {
                    "mode": str(mode_value),
                    "count": mode_count,
                    "percent": round((mode_count / total_rows * 100) if total_rows else 0.0, 2)
                }
                top_vals = []
                for value, count in value_counts.head(top_n).items():
                    top_vals.append({
                        "value": str(value),
                        "count": int(count),
                        "percent": round((count / total_rows * 100) if total_rows else 0.0, 2)
                    })
                results["column_top_values"][col] = top_vals
            else:
                results["column_mode"][col] = {"mode": None, "count": 0, "percent": 0.0}
                results["column_top_values"][col] = []

            # String length stats
            if not non_null.empty:
                str_vals = non_null.astype(str)
                lengths = str_vals.str.len()
                blank_count = int((str_vals.str.strip() == "").sum())
                results["column_string_length"][col] = {
                    "min": int(lengths.min()),
                    "max": int(lengths.max()),
                    "mean": round(float(lengths.mean()), 2),
                    "median": round(float(lengths.median()), 2),
                    "blank_count": blank_count
                }
            else:
                results["column_string_length"][col] = {
                    "min": None, "max": None, "mean": None, "median": None, "blank_count": 0
                }

            # Entropy (categorical)
            if not non_null.empty:
                probs = non_null.value_counts(normalize=True, dropna=True)
                entropy = float(-(probs * np.log2(probs)).sum())
            else:
                entropy = 0.0
            results["column_entropy"][col] = round(entropy, 4)

            # Attempt to convert the entire column to numbers at once
            # errors='coerce' turns anything that isn't a number into NaN 
            numeric_series = pd.to_numeric(col_series, errors='coerce')

            # A. Find Non-Numerics
            is_non_numeric = numeric_series.isna() & col_series.notna()
            unique_text = col_series[is_non_numeric].unique()
            results["column_non_numerics"][col] = unique_text.tolist()
            results["column_non_numeric_count"][col] = int(is_non_numeric.sum())

            # B. Get Statistics (Min/Max/Mean)
            # use the 'numeric_series' which has successfully converted numbers
            # dropna() removes NaN
            valid_numbers = numeric_series.dropna()
            results["column_numeric_count"][col] = int(valid_numbers.count())
            
            if not valid_numbers.empty:
                numeric_columns.append(col)
                numeric_data[col] = numeric_series
                count = int(valid_numbers.count())
                mean_val = float(valid_numbers.mean())
                std_val = float(valid_numbers.std(ddof=1)) if count > 1 else 0.0
                var_val = float(valid_numbers.var(ddof=1)) if count > 1 else 0.0
                q1 = float(valid_numbers.quantile(0.25))
                q3 = float(valid_numbers.quantile(0.75))
                iqr = q3 - q1
                p5 = float(valid_numbers.quantile(0.05))
                p95 = float(valid_numbers.quantile(0.95))
                zeros = int((valid_numbers == 0).sum())
                negatives = int((valid_numbers < 0).sum())

                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers_iqr = int(((valid_numbers < lower) | (valid_numbers > upper)).sum())
                else:
                    outliers_iqr = 0

                if std_val > 0:
                    zscores = (valid_numbers - mean_val) / std_val
                    outliers_z = int((np.abs(zscores) > 3).sum())
                else:
                    outliers_z = 0

                results["column_stats"][col] = {
                    "min": float(valid_numbers.min()),
                    "max": float(valid_numbers.max()),
                    "mean": round(mean_val, 2),
                    "median": round(float(valid_numbers.median()), 2),
                    "std": round(std_val, 4),
                    "variance": round(var_val, 4),
                    "q1": round(q1, 4),
                    "q3": round(q3, 4),
                    "iqr": round(iqr, 4),
                    "p5": round(p5, 4),
                    "p95": round(p95, 4),
                    "skew": round(float(valid_numbers.skew()), 4),
                    "kurtosis": round(float(valid_numbers.kurt()), 4),
                    "zeros": zeros,
                    "negatives": negatives,
                    "outliers_iqr": outliers_iqr,
                    "outliers_zscore": outliers_z,
                    "count": count,
                    "count_percent": round((count / total_rows * 100) if total_rows else 0.0, 2)
                }

        # --- 3. RDKit-based stats (SMILES in first column) ---
        results["rdkit"]["available"] = RDKIT_AVAILABLE
        if RDKIT_AVAILABLE and total_cols > 0 and total_rows > 0:
            smiles_series = df.iloc[:, 0].fillna("").astype(str)
            smiles_total = len(smiles_series)
            smiles_empty = int((smiles_series.str.strip() == "").sum())
            invalid_examples = []
            invalid_examples_limit = 10

            descriptor_funcs = {
                "MolWt": Descriptors.MolWt,
                "LogP": Crippen.MolLogP,
                "TPSA": Descriptors.TPSA,
                "HBD": Lipinski.NumHDonors,
                "HBA": Lipinski.NumHAcceptors,
                "RotB": Lipinski.NumRotatableBonds,
                "Rings": Lipinski.RingCount,
                "AromaticRings": Lipinski.NumAromaticRings,
                "HeavyAtoms": Lipinski.HeavyAtomCount,
                "HeteroAtoms": Lipinski.NumHeteroatoms,
                "FractionCSP3": Lipinski.FractionCSP3,
                "QED": Descriptors.qed,
                "MolMR": Crippen.MolMR,
            }

            descriptor_values = {key: [] for key in descriptor_funcs.keys()}
            descriptor_failures = {key: 0 for key in descriptor_funcs.keys()}

            valid_count = 0
            parse_failures = 0
            exception_count = 0

            for smi in smiles_series:
                smi_str = smi.strip()
                if not smi_str:
                    continue
                try:
                    mol = Chem.MolFromSmiles(smi_str)
                except Exception:
                    exception_count += 1
                    if len(invalid_examples) < invalid_examples_limit:
                        invalid_examples.append(smi_str)
                    continue

                if mol is None:
                    parse_failures += 1
                    if len(invalid_examples) < invalid_examples_limit:
                        invalid_examples.append(smi_str)
                    continue

                valid_count += 1
                for name, func in descriptor_funcs.items():
                    try:
                        descriptor_values[name].append(float(func(mol)))
                    except Exception:
                        descriptor_failures[name] += 1

            invalid_count = smiles_total - valid_count
            invalid_percent = round((invalid_count / smiles_total * 100) if smiles_total else 0.0, 2)

            def summarize_numeric(values):
                if not values:
                    return {"count": 0}
                series = pd.Series(values, dtype=float).dropna()
                if series.empty:
                    return {"count": 0}
                return {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": round(float(series.mean()), 2),
                    "median": round(float(series.median()), 2),
                    "std": round(float(series.std(ddof=1)) if len(series) > 1 else 0.0, 4),
                    "q1": round(float(series.quantile(0.25)), 4),
                    "q3": round(float(series.quantile(0.75)), 4),
                    "p5": round(float(series.quantile(0.05)), 4),
                    "p95": round(float(series.quantile(0.95)), 4),
                    "count": int(series.count()),
                }

            descriptor_stats = {}
            for name, values in descriptor_values.items():
                stats = summarize_numeric(values)
                stats["missing_count"] = max(valid_count - stats.get("count", 0), 0)
                descriptor_stats[name] = stats

            results["rdkit"].update({
                "smiles_column": df.columns[0],
                "total_smiles": smiles_total,
                "empty_or_whitespace": smiles_empty,
                "valid_smiles": valid_count,
                "invalid_smiles": invalid_count,
                "invalid_smiles_percent": invalid_percent,
                "parse_failures": parse_failures,
                "rdkit_exceptions": exception_count,
                "invalid_smiles_examples": invalid_examples,
                "descriptor_failures": descriptor_failures,
                "descriptor_stats": descriptor_stats
            })
        else:
            results["rdkit"]["error"] = "RDKit not installed or no SMILES column"

        # --- 4. Correlations (Numeric Columns) ---
        if numeric_columns:
            numeric_df = pd.DataFrame(numeric_data)
            if len(numeric_columns) >= 2:
                pearson_corr = numeric_df.corr(method="pearson")
                spearman_corr = numeric_df.corr(method="spearman")
                results["correlations"]["pearson"] = pearson_corr.round(4).to_dict()
                results["correlations"]["spearman"] = spearman_corr.round(4).to_dict()

                high_pairs_pearson = []
                cols = pearson_corr.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        value = pearson_corr.iloc[i, j]
                        if not np.isnan(value) and abs(value) >= corr_threshold:
                            high_pairs_pearson.append({
                                "col_a": cols[i],
                                "col_b": cols[j],
                                "corr": round(float(value), 4)
                            })
                results["correlations"]["high_pairs_pearson"] = high_pairs_pearson

                high_pairs_spearman = []
                cols = spearman_corr.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        value = spearman_corr.iloc[i, j]
                        if not np.isnan(value) and abs(value) >= corr_threshold:
                            high_pairs_spearman.append({
                                "col_a": cols[i],
                                "col_b": cols[j],
                                "corr": round(float(value), 4)
                            })
                results["correlations"]["high_pairs_spearman"] = high_pairs_spearman

        # --- 5. sklearn-based stats (optional) ---
        if SKLEARN_AVAILABLE and numeric_columns and total_rows > 0:
            numeric_df = pd.DataFrame(numeric_data)
            X = numeric_df.copy()
            for col in X.columns:
                X[col] = X[col].fillna(X[col].mean())
            X = X.fillna(0.0)

            results["sklearn"]["available"] = True

            # VarianceThreshold for near-constant columns
            variance_threshold = 1e-12
            if X.shape[0] >= 1 and X.shape[1] >= 1:
                vt = VarianceThreshold(threshold=variance_threshold)
                vt.fit(X)
                support = vt.get_support()
                kept_cols = list(np.array(X.columns)[support])
                removed_cols = [col for col in X.columns if col not in kept_cols]
                results["sklearn"]["variance_threshold"] = {
                    "threshold": variance_threshold,
                    "kept_columns": kept_cols,
                    "removed_columns": removed_cols
                }

            # PCA explained variance (standardized)
            if X.shape[1] >= 1 and X.shape[0] >= 2:
                n_components = min(10, X.shape[1], X.shape[0])
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                pca = PCA(n_components=n_components)
                pca.fit(X_scaled)
                results["sklearn"]["pca"] = {
                    "n_components": int(n_components),
                    "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
                    "explained_variance_ratio_cumulative": [
                        round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)
                    ]
                }

            # Mutual information vs target column (if provided)
            if target_column and target_column in df.columns:
                X_feat = X.drop(columns=[target_column], errors="ignore")
                if X_feat.shape[1] >= 1:
                    target_series = df[target_column]
                    target_numeric = pd.to_numeric(target_series, errors='coerce')
                    numeric_ratio = target_numeric.notna().mean() if total_rows else 0.0
                    try:
                        if numeric_ratio > 0.9 and target_numeric.nunique(dropna=True) > 20:
                            y = target_numeric.fillna(target_numeric.mean())
                            mi = mutual_info_regression(X_feat, y)
                            mi_type = "regression"
                        else:
                            y = target_series.fillna("").astype(str)
                            y_codes, _ = pd.factorize(y)
                            mi = mutual_info_classif(X_feat, y_codes, discrete_features=False)
                            mi_type = "classification"
                        mi_scores = sorted(
                            [
                                {"column": col, "mi": round(float(score), 6)}
                                for col, score in zip(X_feat.columns, mi)
                            ],
                            key=lambda item: item["mi"],
                            reverse=True
                        )
                        results["sklearn"]["mutual_information"] = {
                            "target_column": target_column,
                            "type": mi_type,
                            "scores": mi_scores
                        }
                    except Exception as e:
                        results["sklearn"]["mutual_information"] = {
                            "target_column": target_column,
                            "error": str(e)
                        }
        else:
            results["sklearn"]["available"] = False
            results["sklearn"]["error"] = "scikit-learn not installed or no numeric columns"

        # --- 6. Save to JSON ---
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
            
        print(f"Pandas Analysis complete. Saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    if len(sys.argv) > 2:
        output_json = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        output_json = f"eda_{base_name}.json"
    target_column = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("EDA_TARGET_COLUMN", "smiles")
    if os.path.exists(input_csv):
        analyze_csv_pandas(input_csv, output_json, target_column=target_column)
    else:
        print(f"File {input_csv} not found.")
