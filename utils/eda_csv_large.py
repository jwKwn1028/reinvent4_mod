#!/usr/bin/env python3
"""
csv_eda_large.py

Large-CSV EDA (2M+ rows) using streaming/chunked pandas.
Designed to run on a laptop (no full-file DataFrame in memory).

Compared to the original csv_eda.py:
- Uses chunked reading (pd.read_csv(..., chunksize=...))
- Computes numeric stats via online moments (no full numeric matrix kept)
- Computes quantiles/entropy/top-values via sampling/approximation
- Optional 2nd pass for outlier counts (IQR + z-score) + correlation/sklearn on a sample
- RDKit section is optional + sampled (doing RDKit for all 2M rows can be very slow)

Dependencies:
  - pandas, numpy
  - scikit-learn (optional)
  - rdkit (optional)

Example:
  python csv_eda_large.py big.csv eda_big.json --smiles-col SMILES --target-column y --chunksize 100000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# Optional deps
# ----------------------------
try:
    from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors, Lipinski, Crippen
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


# ----------------------------
# Helpers
# ----------------------------
def _as_u64(h: int) -> int:
    return h & ((1 << 64) - 1)


class ReservoirSampler:
    """Uniform reservoir sampler for streaming data."""

    __slots__ = ("k", "n_seen", "samples", "rng")

    def __init__(self, k: int, seed: int = 0) -> None:
        self.k = int(k)
        self.n_seen = 0
        self.samples: List[Any] = []
        self.rng = random.Random(seed)

    def update(self, x: Any) -> None:
        if self.k <= 0:
            return
        self.n_seen += 1
        if len(self.samples) < self.k:
            self.samples.append(x)
            return
        j = self.rng.randrange(self.n_seen)
        if j < self.k:
            self.samples[j] = x

    def update_many(self, xs: np.ndarray) -> None:
        # xs should be 1D
        for x in xs:
            self.update(x)


class KMVSketch:
    """
    K-minimum values sketch for approximate distinct count.
    Stores k smallest 64-bit hashes.
    """

    __slots__ = ("k", "heap", "n_added")

    def __init__(self, k: int) -> None:
        self.k = int(k)
        self.heap: List[int] = []  # max-heap via negative values
        self.n_added = 0

    def add_hash_u64(self, hu64: int) -> None:
        if self.k <= 0:
            return
        self.n_added += 1
        if len(self.heap) < self.k:
            # store negative for max-heap behavior
            import heapq

            heapq.heappush(self.heap, -int(hu64))
            return

        # if new hash is smaller than current largest among kept k
        largest_kept = -self.heap[0]
        if hu64 < largest_kept:
            import heapq

            heapq.heapreplace(self.heap, -int(hu64))

    def estimate(self) -> Optional[float]:
        if self.k <= 0:
            return None
        if len(self.heap) < self.k:
            # not enough data; exact distinct could be <= n_added but we didn't store exact
            # return a conservative lower bound
            return float(len(self.heap)) if self.heap else 0.0

        kth_smallest = -max(self.heap)  # because heap stores -hash; max(-hash) is kth smallest?
        # Actually our heap stores k items; the *largest* kept hash is -heap[0].
        kth_smallest = -self.heap[0]

        # Convert to [0,1)
        r = (kth_smallest + 1) / float(1 << 64)
        if r <= 0.0:
            return None
        # Classic KMV estimator: (k-1)/R_k
        return float((self.k - 1) / r)


@dataclass
class OnlineMoments:
    """Online moments up to 4th (for mean/var/skew/kurtosis) + min/max."""

    n: int = 0
    mean: float = 0.0
    M2: float = 0.0
    M3: float = 0.0
    M4: float = 0.0
    vmin: float = math.inf
    vmax: float = -math.inf
    zeros: int = 0
    negatives: int = 0

    def update_many(self, x: np.ndarray) -> None:
        # x should be 1D float array with no NaN
        for val in x:
            self.update(float(val))

    def update(self, x: float) -> None:
        n1 = self.n
        self.n += 1

        delta = x - self.mean
        delta_n = delta / self.n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1

        self.mean += delta_n
        self.M4 += (
            term1 * delta_n2 * (self.n * self.n - 3 * self.n + 3)
            + 6 * delta_n2 * self.M2
            - 4 * delta_n * self.M3
        )
        self.M3 += term1 * delta_n * (self.n - 2) - 3 * delta_n * self.M2
        self.M2 += term1

        if x < self.vmin:
            self.vmin = x
        if x > self.vmax:
            self.vmax = x
        if x == 0.0:
            self.zeros += 1
        if x < 0.0:
            self.negatives += 1

    def finalize(self) -> Dict[str, Any]:
        if self.n <= 0:
            return {"count": 0}

        if self.n > 1:
            var = self.M2 / (self.n - 1)
            std = math.sqrt(var) if var >= 0 else 0.0
        else:
            var = 0.0
            std = 0.0

        # Skew/kurtosis (excess) as in pandas by default? We’ll provide conventional moment-based.
        if self.n > 2 and self.M2 != 0.0:
            skew = math.sqrt(self.n) * self.M3 / (self.M2 ** 1.5)
        else:
            skew = 0.0

        if self.n > 3 and self.M2 != 0.0:
            kurt = (self.n * self.M4) / (self.M2 * self.M2) - 3.0
        else:
            kurt = 0.0

        return {
            "min": float(self.vmin) if self.vmin != math.inf else None,
            "max": float(self.vmax) if self.vmax != -math.inf else None,
            "mean": float(self.mean),
            "std": float(std),
            "variance": float(var),
            "skew": float(skew),
            "kurtosis": float(kurt),
            "zeros": int(self.zeros),
            "negatives": int(self.negatives),
            "count": int(self.n),
        }


@dataclass
class ColumnEDAState:
    missing: int = 0
    non_null: int = 0

    # string length stats
    strlen_min: int = 10**9
    strlen_max: int = 0
    strlen_sum: int = 0
    blank_count: int = 0

    # approximate top values via counters (pruned)
    top_counter: Counter = field(default_factory=Counter)

    # approximate distinct count
    kmv: Optional[KMVSketch] = None

    # entropy sample
    entropy_sample: Optional[ReservoirSampler] = None

    # numeric parsing stats
    numeric_count: int = 0
    non_numeric_count: int = 0
    non_numeric_examples: List[str] = field(default_factory=list)
    numeric_moments: OnlineMoments = field(default_factory=OnlineMoments)
    numeric_sample: Optional[ReservoirSampler] = None  # for quantiles

    # second pass counts
    outliers_iqr: int = 0
    outliers_zscore: int = 0

    def update_string_stats(self, s: pd.Series) -> None:
        # s: non-null strings (dtype object)
        sv = s.astype(str)
        stripped = sv.str.strip()
        blanks = (stripped == "").sum()
        self.blank_count += int(blanks)

        lengths = sv.str.len().to_numpy(dtype=np.int64, copy=False)
        if lengths.size:
            self.strlen_min = min(self.strlen_min, int(lengths.min()))
            self.strlen_max = max(self.strlen_max, int(lengths.max()))
            self.strlen_sum += int(lengths.sum())

    def push_top_values(self, s: pd.Series, top_per_chunk: int, global_cap: int) -> None:
        if s.empty:
            return
        vc = s.value_counts(dropna=True).head(top_per_chunk)
        # Update global counter
        for k, v in vc.items():
            self.top_counter[str(k)] += int(v)
        # Prune counter to cap
        if global_cap > 0 and len(self.top_counter) > global_cap:
            self.top_counter = Counter(dict(self.top_counter.most_common(global_cap)))

    def add_entropy_samples(self, s: pd.Series) -> None:
        if self.entropy_sample is None or s.empty:
            return
        for v in s.astype(str).tolist():
            self.entropy_sample.update(v)

    def add_kmv(self, s: pd.Series) -> None:
        if self.kmv is None or s.empty:
            return
        # Use built-in hash (fast). If you want stable estimates across runs, set PYTHONHASHSEED.
        for v in s.astype(str).tolist():
            self.kmv.add_hash_u64(_as_u64(hash(v)))

    def add_numeric(self, numeric_vals: np.ndarray) -> None:
        if numeric_vals.size == 0:
            return
        self.numeric_moments.update_many(numeric_vals)
        self.numeric_count += int(numeric_vals.size)
        if self.numeric_sample is not None:
            self.numeric_sample.update_many(numeric_vals)


def _safe_float_array(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    arr = arr[~np.isnan(arr)]
    return arr


def _entropy_from_samples(samples: List[str]) -> float:
    if not samples:
        return 0.0
    c = Counter(samples)
    n = sum(c.values())
    if n <= 0:
        return 0.0
    ent = 0.0
    for cnt in c.values():
        p = cnt / n
        ent -= p * math.log2(p)
    return float(ent)


def _quantiles_from_numeric_sample(samples: List[float]) -> Dict[str, Any]:
    if not samples:
        return {"q1": None, "q3": None, "p5": None, "p95": None, "median": None, "iqr": None}
    a = np.asarray(samples, dtype=np.float64)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"q1": None, "q3": None, "p5": None, "p95": None, "median": None, "iqr": None}
    q1 = float(np.quantile(a, 0.25))
    q3 = float(np.quantile(a, 0.75))
    p5 = float(np.quantile(a, 0.05))
    p95 = float(np.quantile(a, 0.95))
    med = float(np.quantile(a, 0.50))
    return {"q1": q1, "q3": q3, "p5": p5, "p95": p95, "median": med, "iqr": q3 - q1}


def _infer_numeric_columns(states: Dict[str, ColumnEDAState], ratio_threshold: float, min_count: int) -> List[str]:
    numeric_cols: List[str] = []
    for col, st in states.items():
        if st.non_null <= 0:
            continue
        ratio = st.numeric_count / max(st.non_null, 1)
        if st.numeric_count >= min_count and ratio >= ratio_threshold:
            numeric_cols.append(col)
    return numeric_cols


# ----------------------------
# RDKit streaming stats (sampled)
# ----------------------------
@dataclass
class RDKitStatsState:
    max_mols: int
    sample_rate: float
    seed: int

    total_smiles: int = 0
    empty_or_whitespace: int = 0
    valid_smiles: int = 0
    invalid_smiles: int = 0
    parse_failures: int = 0
    rdkit_exceptions: int = 0
    invalid_examples: List[str] = field(default_factory=list)

    descriptor_moments: Dict[str, OnlineMoments] = field(default_factory=dict)
    descriptor_samples: Dict[str, ReservoirSampler] = field(default_factory=dict)
    descriptor_failures: Dict[str, int] = field(default_factory=dict)

    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

        # Keep it aligned with your original set + scaffold
        self.descriptor_funcs = {
            "MurckoScaffold": None,  # special
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

        for k in self.descriptor_funcs.keys():
            if k == "MurckoScaffold":
                continue
            self.descriptor_moments[k] = OnlineMoments()
            self.descriptor_samples[k] = ReservoirSampler(k=20000, seed=self.seed + 13)
            self.descriptor_failures[k] = 0

        self.scaffold_samples = ReservoirSampler(k=20000, seed=self.seed + 23)

    def maybe_process_smiles(self, smi: str) -> None:
        if self.valid_smiles >= self.max_mols:
            return
        if self.rng.random() > self.sample_rate:
            return

        smi_str = smi.strip()
        if not smi_str:
            return

        try:
            mol = Chem.MolFromSmiles(smi_str)
        except Exception:
            self.rdkit_exceptions += 1
            if len(self.invalid_examples) < 10:
                self.invalid_examples.append(smi_str)
            return

        if mol is None:
            self.parse_failures += 1
            if len(self.invalid_examples) < 10:
                self.invalid_examples.append(smi_str)
            return

        self.valid_smiles += 1

        # Scaffold
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaf = ""
        self.scaffold_samples.update(scaf)

        for name, func in self.descriptor_funcs.items():
            if name == "MurckoScaffold":
                continue
            try:
                val = float(func(mol))
                self.descriptor_moments[name].update(val)
                self.descriptor_samples[name].update(val)
            except Exception:
                self.descriptor_failures[name] += 1


def _rdkit_disable_logs() -> None:
    if not RDKIT_AVAILABLE:
        return
    RDLogger.DisableLog("rdApp.*")


# ----------------------------
# Main analysis
# ----------------------------
def analyze_csv_large(
    input_file: str,
    output_file: str,
    target_column: Optional[str],
    smiles_col: Optional[str],
    *,
    chunksize: int = 100_000,
    top_n: int = 5,
    corr_threshold: float = 0.9,
    seed: int = 0,
    numeric_ratio_threshold: float = 0.9,
    numeric_min_count: int = 1000,
    # approximation controls
    kmv_k: int = 256,
    entropy_sample_k: int = 50_000,
    numeric_sample_k: int = 200_000,
    top_per_chunk: int = 200,
    top_global_cap: int = 2000,
    # 2nd pass controls
    second_pass: bool = True,
    corr_sample_rows: int = 200_000,
    sklearn_sample_rows: int = 200_000,
    # RDKit controls
    rdkit_enable: bool = True,
    rdkit_sample_rate: float = 0.02,
    rdkit_max_mols: int = 200_000,
) -> None:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input not found: {input_file}")

    rng = random.Random(seed)

    # Stream reader (dtype=str keeps IDs like "001" intact)
    reader = pd.read_csv(input_file, dtype=str, chunksize=chunksize, low_memory=False)

    # We'll initialize states after seeing first chunk (to know columns)
    states: Dict[str, ColumnEDAState] = {}
    headers: List[str] = []

    # File-level metadata (streamed)
    total_rows = 0
    total_cols = 0
    missing_cells = 0

    # Approximate duplicates in first column via hash set (may have rare collisions)
    first_col_name: Optional[str] = None
    first_col_seen_hashes: set[int] = set()
    first_col_duplicates_examples: List[str] = []
    first_col_duplicates_set: set[str] = set()

    # Row duplicates: exact is expensive; we provide optional approximate if needed (disabled here)
    duplicate_rows_estimate: Optional[int] = None

    # RDKit sampled stats
    rdkit_state: Optional[RDKitStatsState] = None
    if rdkit_enable and RDKIT_AVAILABLE:
        _rdkit_disable_logs()
        rdkit_state = RDKitStatsState(
            max_mols=int(rdkit_max_mols),
            sample_rate=float(rdkit_sample_rate),
            seed=int(seed),
        )

    # ----------------------------
    # PASS 1: column basics + numeric moments + samples
    # ----------------------------
    for chunk_idx, df in enumerate(reader):
        if chunk_idx == 0:
            headers = df.columns.tolist()
            total_cols = len(headers)
            if total_cols == 0:
                raise ValueError("CSV has no columns.")
            first_col_name = headers[0]

            # setup per-column states
            for col in headers:
                st = ColumnEDAState()
                st.kmv = KMVSketch(kmv_k) if kmv_k > 0 else None
                st.entropy_sample = ReservoirSampler(entropy_sample_k, seed=seed + 101) if entropy_sample_k > 0 else None
                st.numeric_sample = ReservoirSampler(numeric_sample_k, seed=seed + 202) if numeric_sample_k > 0 else None
                states[col] = st

            # Default smiles column
            if smiles_col is None:
                smiles_col = first_col_name

        # Count missing cells
        missing_cells += int(df.isna().sum().sum())
        n_rows = len(df)
        total_rows += n_rows

        # First col duplicate examples (hash-based)
        if first_col_name is not None:
            col0 = df[first_col_name].fillna("").astype(str)
            for v in col0.tolist():
                vv = v.strip()
                if not vv:
                    continue
                hv = _as_u64(hash(vv))
                if hv in first_col_seen_hashes:
                    if vv not in first_col_duplicates_set and len(first_col_duplicates_examples) < 1000:
                        first_col_duplicates_set.add(vv)
                        first_col_duplicates_examples.append(vv)
                else:
                    first_col_seen_hashes.add(hv)

        # RDKit sampling
        if rdkit_state is not None and smiles_col in df.columns:
            sers = df[smiles_col].fillna("").astype(str)
            rdkit_state.total_smiles += int(len(sers))
            rdkit_state.empty_or_whitespace += int((sers.str.strip() == "").sum())
            for smi in sers.tolist():
                if rdkit_state.valid_smiles >= rdkit_state.max_mols:
                    break
                rdkit_state.maybe_process_smiles(smi)

        # Per-column processing
        for col in headers:
            st = states[col]
            col_series = df[col]

            # Missing/non-null
            nn = col_series.dropna()
            st.non_null += int(len(nn))
            st.missing += int(n_rows - len(nn))

            if nn.empty:
                continue

            # String stats (lengths, blanks)
            st.update_string_stats(nn)

            # Samples for entropy + approx distinct
            st.add_entropy_samples(nn)
            st.add_kmv(nn)

            # Top values (approx, pruned)
            st.push_top_values(nn.astype(str), top_per_chunk=top_per_chunk, global_cap=top_global_cap)

            # Numeric parse stats
            numeric_series = pd.to_numeric(col_series, errors="coerce")
            is_non_numeric = numeric_series.isna() & col_series.notna()
            nonnum_count = int(is_non_numeric.sum())
            st.non_numeric_count += nonnum_count

            # Keep a few example non-numeric tokens
            if nonnum_count > 0 and len(st.non_numeric_examples) < 50:
                uniques = col_series[is_non_numeric].astype(str).unique().tolist()
                for u in uniques:
                    if u not in st.non_numeric_examples:
                        st.non_numeric_examples.append(u)
                        if len(st.non_numeric_examples) >= 50:
                            break

            valid_numbers = numeric_series.dropna()
            if not valid_numbers.empty:
                arr = valid_numbers.to_numpy(dtype=np.float64, copy=False)
                st.add_numeric(arr)

        if (chunk_idx + 1) % 10 == 0:
            print(f"[PASS1] chunks={chunk_idx+1} rows={total_rows}", file=sys.stderr)

    total_cells = int(total_rows * total_cols)

    # Determine numeric columns to use for corr/sklearn/outliers
    numeric_columns = _infer_numeric_columns(
        states,
        ratio_threshold=float(numeric_ratio_threshold),
        min_count=int(numeric_min_count),
    )

    # Finalize numeric quantiles from samples (pass1)
    numeric_quantiles: Dict[str, Dict[str, Any]] = {}
    for col in numeric_columns:
        st = states[col]
        samples = st.numeric_sample.samples if st.numeric_sample is not None else []
        numeric_quantiles[col] = _quantiles_from_numeric_sample(samples)

    # ----------------------------
    # PASS 2: outlier counts + correlation/sklearn samples (optional)
    # ----------------------------
    corr_sample: Optional[ReservoirSampler] = None
    sklearn_sample: Optional[ReservoirSampler] = None

    # Store sampled rows as dicts or arrays; arrays are faster/more compact
    corr_rows: List[List[float]] = []
    sklearn_rows: List[List[float]] = []
    sklearn_target: List[Any] = []

    if second_pass and numeric_columns:
        # Re-open reader
        reader2 = pd.read_csv(input_file, dtype=str, chunksize=chunksize, low_memory=False)

        corr_sampler = ReservoirSampler(corr_sample_rows, seed=seed + 303) if corr_sample_rows > 0 else None
        skl_sampler = ReservoirSampler(sklearn_sample_rows, seed=seed + 404) if sklearn_sample_rows > 0 else None

        # Precompute mean/std for z-score from pass1 moments
        mean_std: Dict[str, Tuple[float, float]] = {}
        for col in numeric_columns:
            fin = states[col].numeric_moments.finalize()
            mean_std[col] = (float(fin.get("mean", 0.0)), float(fin.get("std", 0.0)))

        for chunk_idx, df in enumerate(reader2):
            # Numeric frame for numeric_columns only
            num_df = pd.DataFrame(index=df.index)
            for col in numeric_columns:
                num_df[col] = pd.to_numeric(df[col], errors="coerce")

            # Outlier counts (IQR + z-score) per column
            for col in numeric_columns:
                st = states[col]
                s = num_df[col].dropna()
                if s.empty:
                    continue
                a = s.to_numpy(dtype=np.float64, copy=False)

                q = numeric_quantiles.get(col, {})
                q1 = q.get("q1", None)
                q3 = q.get("q3", None)
                iqr = q.get("iqr", None)
                if q1 is not None and q3 is not None and iqr is not None and iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    st.outliers_iqr += int(((a < lower) | (a > upper)).sum())

                mu, sd = mean_std.get(col, (0.0, 0.0))
                if sd > 0.0:
                    z = (a - mu) / sd
                    st.outliers_zscore += int((np.abs(z) > 3).sum())

            # Correlation/sklearn sampling (sample rows; fill NAs)
            if corr_sampler is not None or skl_sampler is not None:
                # Fill with column means for sampling matrix
                filled = num_df.copy()
                for col in numeric_columns:
                    mu, _ = mean_std[col]
                    filled[col] = filled[col].fillna(mu)
                filled = filled.fillna(0.0)

                # Row-wise reservoir sampling (store list[float])
                # Use indices to avoid building Python lists for all rows
                for i in range(len(filled)):
                    row_vals = filled.iloc[i].to_numpy(dtype=np.float64, copy=False).tolist()

                    if corr_sampler is not None:
                        corr_sampler.update(row_vals)
                    if skl_sampler is not None:
                        skl_sampler.update(row_vals)
                        if target_column and target_column in df.columns:
                            sklearn_target.append(df.iloc[i][target_column])

            if (chunk_idx + 1) % 10 == 0:
                print(f"[PASS2] chunks={chunk_idx+1}", file=sys.stderr)

        # Materialize sampled rows
        if corr_sampler is not None:
            corr_rows = corr_sampler.samples  # list[list[float]]
        if skl_sampler is not None:
            sklearn_rows = skl_sampler.samples  # list[list[float]]

    # ----------------------------
    # Build results JSON
    # ----------------------------
    results: Dict[str, Any] = {
        "headers": headers,
        "metadata": {
            "total_rows": int(total_rows),
            "columns": int(total_cols),
            "total_cells": int(total_cells),
            "missing_cells": int(missing_cells),
            "missing_percent": round((missing_cells / total_cells * 100) if total_cells else 0.0, 4),
            "duplicate_rows": duplicate_rows_estimate,  # not computed exactly for large CSV
            "notes": [
                "This is a large-CSV streaming EDA report.",
                "Some fields are approximate (unique counts, entropy, top values, correlations, sklearn) and based on sketches/samples.",
                "For stable hashing across runs, set PYTHONHASHSEED (otherwise Python string hashes are randomized per process).",
            ],
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
        "first_column_duplicates": {
            "column": first_col_name,
            "examples": first_col_duplicates_examples,
            "note": "Hash-based detection; rare collisions possible. Examples list is capped.",
        },
    }

    # Column-level fields
    for col in headers:
        st = states[col]

        # Missing and unique
        results["column_missing"][col] = {
            "missing": int(st.missing),
            "missing_percent": round((st.missing / total_rows * 100) if total_rows else 0.0, 4),
        }

        unique_est = st.kmv.estimate() if st.kmv is not None else None
        results["column_unique"][col] = {
            "unique_estimate": None if unique_est is None else int(round(unique_est)),
            "unique_estimate_method": f"KMV(k={kmv_k})" if st.kmv is not None else None,
            "unique_percent_estimate": (
                round((unique_est / total_rows * 100), 4) if (unique_est is not None and total_rows) else None
            ),
        }

        # Mode + top values (approx)
        if st.top_counter:
            mode_val, mode_cnt = st.top_counter.most_common(1)[0]
            results["column_mode"][col] = {
                "mode": str(mode_val),
                "count_estimate": int(mode_cnt),
                "percent_estimate": round((mode_cnt / total_rows * 100) if total_rows else 0.0, 4),
                "note": "Estimated from pruned counters; may be approximate.",
            }
            top_vals = []
            for v, c in st.top_counter.most_common(top_n):
                top_vals.append(
                    {
                        "value": str(v),
                        "count_estimate": int(c),
                        "percent_estimate": round((c / total_rows * 100) if total_rows else 0.0, 4),
                    }
                )
            results["column_top_values"][col] = top_vals
        else:
            results["column_mode"][col] = {"mode": None, "count_estimate": 0, "percent_estimate": 0.0}
            results["column_top_values"][col] = []

        # String length stats
        if st.non_null > 0:
            mean_len = st.strlen_sum / max(st.non_null, 1)
            results["column_string_length"][col] = {
                "min": None if st.strlen_min == 10**9 else int(st.strlen_min),
                "max": int(st.strlen_max),
                "mean": round(float(mean_len), 4),
                "blank_count": int(st.blank_count),
            }
        else:
            results["column_string_length"][col] = {"min": None, "max": None, "mean": None, "blank_count": 0}

        # Entropy (sample-based)
        ent = 0.0
        if st.entropy_sample is not None:
            ent = _entropy_from_samples(st.entropy_sample.samples)
        results["column_entropy"][col] = {
            "entropy_estimate": round(float(ent), 6),
            "method": f"reservoir_sample(k={entropy_sample_k})" if st.entropy_sample is not None else None,
        }

        # Non-numerics
        results["column_non_numerics"][col] = st.non_numeric_examples
        results["column_non_numeric_count"][col] = int(st.non_numeric_count)
        results["column_numeric_count"][col] = int(st.numeric_count)

        # Numeric stats
        if col in numeric_columns:
            fin = st.numeric_moments.finalize()
            qs = numeric_quantiles.get(col, {})
            # Merge + add outlier counts (from pass2 if enabled)
            results["column_stats"][col] = {
                **{
                    "min": fin.get("min"),
                    "max": fin.get("max"),
                    "mean": round(float(fin.get("mean", 0.0)), 6),
                    "median_estimate": None if qs.get("median") is None else round(float(qs["median"]), 6),
                    "std": round(float(fin.get("std", 0.0)), 6),
                    "variance": round(float(fin.get("variance", 0.0)), 6),
                    "q1_estimate": None if qs.get("q1") is None else round(float(qs["q1"]), 6),
                    "q3_estimate": None if qs.get("q3") is None else round(float(qs["q3"]), 6),
                    "iqr_estimate": None if qs.get("iqr") is None else round(float(qs["iqr"]), 6),
                    "p5_estimate": None if qs.get("p5") is None else round(float(qs["p5"]), 6),
                    "p95_estimate": None if qs.get("p95") is None else round(float(qs["p95"]), 6),
                    "skew": round(float(fin.get("skew", 0.0)), 6),
                    "kurtosis": round(float(fin.get("kurtosis", 0.0)), 6),
                    "zeros": int(fin.get("zeros", 0)),
                    "negatives": int(fin.get("negatives", 0)),
                    "count": int(fin.get("count", 0)),
                    "count_percent": round((fin.get("count", 0) / total_rows * 100) if total_rows else 0.0, 4),
                },
                "outliers_iqr_estimate_count": int(st.outliers_iqr) if second_pass else None,
                "outliers_zscore_estimate_count": int(st.outliers_zscore) if second_pass else None,
                "notes": [
                    f"Quantiles are estimated from reservoir_sample(k={numeric_sample_k}).",
                    "Outlier counts require second pass; counts are based on estimated quantiles / final mean/std.",
                ],
            }

    # ----------------------------
    # Correlations (sample-based)
    # ----------------------------
    if corr_rows and len(numeric_columns) >= 2:
        corr_arr = np.asarray(corr_rows, dtype=np.float64)
        corr_df = pd.DataFrame(corr_arr, columns=numeric_columns)
        pearson = corr_df.corr(method="pearson")
        spearman = corr_df.corr(method="spearman")

        results["correlations"]["available"] = True
        results["correlations"]["method"] = f"sampled_rows(reservoir k={corr_sample_rows})"
        results["correlations"]["pearson"] = pearson.round(4).to_dict()
        results["correlations"]["spearman"] = spearman.round(4).to_dict()

        high_pairs_pearson = []
        cols = pearson.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = pearson.iloc[i, j]
                if not np.isnan(v) and abs(v) >= corr_threshold:
                    high_pairs_pearson.append({"col_a": cols[i], "col_b": cols[j], "corr": round(float(v), 4)})
        results["correlations"]["high_pairs_pearson"] = high_pairs_pearson

        high_pairs_spearman = []
        cols = spearman.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = spearman.iloc[i, j]
                if not np.isnan(v) and abs(v) >= corr_threshold:
                    high_pairs_spearman.append({"col_a": cols[i], "col_b": cols[j], "corr": round(float(v), 4)})
        results["correlations"]["high_pairs_spearman"] = high_pairs_spearman
    else:
        results["correlations"]["available"] = False
        results["correlations"]["error"] = "No correlation sample (second pass disabled, or no numeric columns)."

    # ----------------------------
    # sklearn (sample-based)
    # ----------------------------
    if SKLEARN_AVAILABLE and sklearn_rows and len(numeric_columns) >= 1:
        X = np.asarray(sklearn_rows, dtype=np.float64)
        results["sklearn"]["available"] = True
        results["sklearn"]["method"] = f"sampled_rows(reservoir k={sklearn_sample_rows})"

        # Variance threshold
        vt_threshold = 1e-12
        try:
            vt = VarianceThreshold(threshold=vt_threshold)
            vt.fit(X)
            support = vt.get_support()
            kept_cols = [c for c, ok in zip(numeric_columns, support) if ok]
            removed_cols = [c for c, ok in zip(numeric_columns, support) if not ok]
            results["sklearn"]["variance_threshold"] = {
                "threshold": vt_threshold,
                "kept_columns": kept_cols,
                "removed_columns": removed_cols,
            }
        except Exception as e:
            results["sklearn"]["variance_threshold"] = {"error": str(e)}

        # PCA (standardized)
        try:
            n_components = int(min(10, X.shape[1], X.shape[0]))
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            pca = PCA(n_components=n_components)
            pca.fit(Xs)
            evr = pca.explained_variance_ratio_
            results["sklearn"]["pca"] = {
                "n_components": n_components,
                "explained_variance_ratio": [round(float(v), 6) for v in evr],
                "explained_variance_ratio_cumulative": [round(float(v), 6) for v in np.cumsum(evr)],
            }
        except Exception as e:
            results["sklearn"]["pca"] = {"error": str(e)}

        # Mutual information vs target (if target is provided and we sampled it)
        if target_column:
            try:
                # If target_column is numeric and has many unique values, treat as regression
                # Otherwise treat as classification by factorizing.
                # Use only as many y samples as we have X samples (sklearn_target may be longer if sampling replaced rows)
                y_raw = sklearn_target[: X.shape[0]] if sklearn_target else []
                if not y_raw:
                    results["sklearn"]["mutual_information"] = {"error": "No sampled target values available."}
                else:
                    y_num = pd.to_numeric(pd.Series(y_raw), errors="coerce")
                    numeric_ratio = float(y_num.notna().mean()) if len(y_num) else 0.0

                    if numeric_ratio > 0.9 and y_num.nunique(dropna=True) > 20:
                        y = y_num.fillna(float(y_num.mean())).to_numpy(dtype=np.float64)
                        mi = mutual_info_regression(X, y)
                        mi_type = "regression"
                    else:
                        y_str = pd.Series(y_raw).fillna("").astype(str)
                        y_codes, _ = pd.factorize(y_str)
                        mi = mutual_info_classif(X, y_codes, discrete_features=False)
                        mi_type = "classification"

                    mi_scores = sorted(
                        [{"column": c, "mi": round(float(s), 8)} for c, s in zip(numeric_columns, mi)],
                        key=lambda d: d["mi"],
                        reverse=True,
                    )
                    results["sklearn"]["mutual_information"] = {
                        "target_column": target_column,
                        "type": mi_type,
                        "scores": mi_scores,
                    }
            except Exception as e:
                results["sklearn"]["mutual_information"] = {"target_column": target_column, "error": str(e)}
    else:
        results["sklearn"]["available"] = False
        results["sklearn"]["error"] = "scikit-learn not installed or no sklearn sample/numeric columns."

    # ----------------------------
    # RDKit summary (sampled)
    # ----------------------------
    results["rdkit"]["available"] = bool(RDKIT_AVAILABLE)
    if rdkit_state is not None:
        # Descriptor summary
        desc_stats: Dict[str, Any] = {}
        for name, mom in rdkit_state.descriptor_moments.items():
            fin = mom.finalize()
            qs = _quantiles_from_numeric_sample(rdkit_state.descriptor_samples[name].samples)
            desc_stats[name] = {
                "min": fin.get("min"),
                "max": fin.get("max"),
                "mean": round(float(fin.get("mean", 0.0)), 6),
                "std": round(float(fin.get("std", 0.0)), 6),
                "median_estimate": None if qs.get("median") is None else round(float(qs["median"]), 6),
                "q1_estimate": None if qs.get("q1") is None else round(float(qs["q1"]), 6),
                "q3_estimate": None if qs.get("q3") is None else round(float(qs["q3"]), 6),
                "count_sampled": int(fin.get("count", 0)),
                "failures": int(rdkit_state.descriptor_failures.get(name, 0)),
            }

        results["rdkit"].update(
            {
                "smiles_column": smiles_col,
                "total_smiles_seen": int(rdkit_state.total_smiles),
                "empty_or_whitespace": int(rdkit_state.empty_or_whitespace),
                "mols_sampled_target_max": int(rdkit_state.max_mols),
                "sample_rate": float(rdkit_state.sample_rate),
                "valid_mols_sampled": int(rdkit_state.valid_smiles),
                "invalid_examples": rdkit_state.invalid_examples,
                "parse_failures": int(rdkit_state.parse_failures),
                "rdkit_exceptions": int(rdkit_state.rdkit_exceptions),
                "descriptor_stats": desc_stats,
                "murcko_scaffold_samples": {
                    "method": "reservoir_sample",
                    "k": len(rdkit_state.scaffold_samples.samples),
                    "examples": rdkit_state.scaffold_samples.samples[:50],
                },
                "note": "RDKit section is sampled. Set --rdkit-sample-rate 1.0 and increase --rdkit-max-mols to analyze more (will be slower).",
            }
        )
    else:
        if rdkit_enable and not RDKIT_AVAILABLE:
            results["rdkit"]["error"] = "RDKit not installed."
        else:
            results["rdkit"]["error"] = "RDKit disabled."

    # Save JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[DONE] Saved EDA JSON: {output_file}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help="Input CSV path")
    p.add_argument("output_json", nargs="?", default=None, help="Output JSON path (default: eda_large_<name>.json)")
    p.add_argument("--target-column", default=None, help="Target column for mutual information (optional)")
    p.add_argument("--smiles-col", default=None, help="SMILES column for RDKit stats (default: first column)")
    p.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk (default: 100000)")
    p.add_argument("--top-n", type=int, default=5, help="Top-N values per column (default: 5)")
    p.add_argument("--corr-threshold", type=float, default=0.9, help="High-correlation threshold (default: 0.9)")
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    # Numeric detection thresholds
    p.add_argument("--numeric-ratio-threshold", type=float, default=0.9, help="Numeric ratio threshold (default: 0.9)")
    p.add_argument("--numeric-min-count", type=int, default=1000, help="Min numeric values to treat column as numeric (default: 1000)")

    # Approx controls
    p.add_argument("--kmv-k", type=int, default=256, help="KMV sketch size for unique estimate (default: 256; 0 disables)")
    p.add_argument("--entropy-sample-k", type=int, default=50_000, help="Reservoir size for entropy estimate (default: 50000; 0 disables)")
    p.add_argument("--numeric-sample-k", type=int, default=200_000, help="Reservoir size for numeric quantiles (default: 200000; 0 disables)")
    p.add_argument("--top-per-chunk", type=int, default=200, help="Value-count head() per chunk for top values (default: 200)")
    p.add_argument("--top-global-cap", type=int, default=2000, help="Prune global top counter per column (default: 2000)")

    # Second pass / sampling
    p.add_argument("--no-second-pass", action="store_true", help="Disable second pass (no outlier counts, corr, sklearn)")
    p.add_argument("--corr-sample-rows", type=int, default=200_000, help="Reservoir rows for correlation (default: 200000; 0 disables)")
    p.add_argument("--sklearn-sample-rows", type=int, default=200_000, help="Reservoir rows for sklearn (default: 200000; 0 disables)")

    # RDKit options
    p.add_argument("--no-rdkit", action="store_true", help="Disable RDKit stats")
    p.add_argument("--rdkit-sample-rate", type=float, default=0.02, help="RDKit sampling probability (default: 0.02)")
    p.add_argument("--rdkit-max-mols", type=int, default=200_000, help="Max RDKit molecules to process (default: 200000)")

    args = p.parse_args()

    input_csv = args.input_csv
    if args.output_json is None:
        base = os.path.splitext(os.path.basename(input_csv))[0]
        output_json = f"eda_large_{base}.json"
    else:
        output_json = args.output_json

    analyze_csv_large(
        input_file=input_csv,
        output_file=output_json,
        target_column=args.target_column,
        smiles_col=args.smiles_col,
        chunksize=args.chunksize,
        top_n=args.top_n,
        corr_threshold=args.corr_threshold,
        seed=args.seed,
        numeric_ratio_threshold=args.numeric_ratio_threshold,
        numeric_min_count=args.numeric_min_count,
        kmv_k=args.kmv_k,
        entropy_sample_k=args.entropy_sample_k,
        numeric_sample_k=args.numeric_sample_k,
        top_per_chunk=args.top_per_chunk,
        top_global_cap=args.top_global_cap,
        second_pass=(not args.no_second_pass),
        corr_sample_rows=args.corr_sample_rows,
        sklearn_sample_rows=args.sklearn_sample_rows,
        rdkit_enable=(not args.no_rdkit),
        rdkit_sample_rate=args.rdkit_sample_rate,
        rdkit_max_mols=args.rdkit_max_mols,
    )


if __name__ == "__main__":
    main()
