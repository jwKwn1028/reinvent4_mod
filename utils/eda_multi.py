#!/usr/bin/env python3
"""
csv_eda.py

Comprehensive CSV EDA for chemical ML/RL logs using:
- pandas / numpy
- RDKit (optional)
- scikit-learn (optional)
- umap-learn (optional, compare-to mode)

Outputs a JSON report, and optionally saves helper CSVs + plots to an output directory.

Main features
-------------
1) Column EDA: missing/unique/mode/top-k/string lengths/entropy
2) Numeric stats: quantiles, IQR outliers, z outliers, skew, kurtosis
3) Correlations: Pearson/Spearman + high-correlation pairs (bounded + guarded)
4) RDKit SMILES (if available): valid/invalid, descriptor stats, scaffolds, InChIKey dup estimate, fp diversity
5) sklearn (if available): variance threshold, PCA explained variance, mutual information vs targets
6) compare-to mode (RDKit+sklearn only):
   - Tanimoto max-sim distributions (A→B, B→A) + hist bins (+ optional plot)
   - RDKit descriptor shift (Wasserstein 1D)
   - KL/JS divergence for descriptor distributions + targets
   - PCA 2D coords (+ optional plot)
   - optional t-SNE 2D (+ optional plot)
   - optional UMAP 2D (+ optional plot)
   - scaffold overlap (Jaccard)

Notes / Improvements vs earlier versions
---------------------------------------
- Treat empty/whitespace strings as missing (configurable).
- More robust numeric column detection using a numeric-parse ratio threshold + min unique numeric values.
- Heuristic to avoid misclassifying leading-zero ID-like columns as numeric.
- Bounds on expensive/large outputs (duplicates examples, correlation pairs, optional omission of full matrices).
- Guards on correlation/embedding computations (min rows/cols, parameter clamping).
- Removes "column_non_numerics" and related non-numeric listing from JSON (per request).

Examples
--------
Single dataset:
  python csv_eda.py data.csv eda.json --smiles-column SMILES --targets Score,"LogD (Lipo) (raw)"

Auto targets (if none provided):
  python csv_eda.py data.csv eda.json --smiles-column SMILES

Compare two datasets:
  python csv_eda.py A.csv eda_A.json --compare-to B.csv --smiles-column SMILES --targets Score --outdir stat/eda_out --umap --tsne
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

# -------------------------
# Optional deps
# -------------------------
try:
    from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif  # type: ignore[import-untyped]
    from sklearn.decomposition import PCA  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
    from sklearn.manifold import TSNE  # type: ignore[import-untyped]
    from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import umap  # type: ignore[import-untyped]
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

try:
    from rdkit import Chem, DataStructs  # type: ignore[import-untyped]
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski  # type: ignore[import-untyped]
    from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore[import-untyped]

    try:
        from rdkit.Chem.inchi import MolToInchiKey  # type: ignore[import-untyped]
        INCHI_AVAILABLE = True
    except Exception:
        INCHI_AVAILABLE = False

    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False
    INCHI_AVAILABLE = False

# If RDKit is present but typing/stubs are incomplete, cast module refs to Any
if RDKIT_AVAILABLE:
    Chem = cast(Any, Chem)
    DataStructs = cast(Any, DataStructs)
    AllChem = cast(Any, AllChem)
    Crippen = cast(Any, Crippen)
    Descriptors = cast(Any, Descriptors)
    Lipinski = cast(Any, Lipinski)
    MurckoScaffold = cast(Any, MurckoScaffold)
    if INCHI_AVAILABLE:
        MolToInchiKey = cast(Any, MolToInchiKey)


DEFAULT_TARGET_CANDIDATES = [
    "Score",
    "smiles",
    "mp",
    "de",
    "Dehydrogenation Enthalpy (raw)",
    "LogD (Lipo) (raw)",
    "Synthetic Accessibility score (raw)",
    "Water Solubility (ESOL) (raw)",
]


# -------------------------
# Generic utilities
# -------------------------
def _dedupe_preserve_order(xs: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _entropy_from_series(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return 0.0
    probs = s.value_counts(normalize=True, dropna=True)
    return float(-(probs * np.log2(probs)).sum())


def _summarize_numeric_array(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"count": 0}
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "count": int(x.size),
    }


def _summarize_numeric(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    s = pd.Series(values, dtype=float).dropna()
    if s.empty:
        return {"count": 0}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": round(float(s.mean()), 6),
        "median": round(float(s.median()), 6),
        "std": round(float(s.std(ddof=1)) if len(s) > 1 else 0.0, 8),
        "q1": round(float(s.quantile(0.25)), 8),
        "q3": round(float(s.quantile(0.75)), 8),
        "p5": round(float(s.quantile(0.05)), 8),
        "p95": round(float(s.quantile(0.95)), 8),
        "count": int(s.count()),
    }


def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Simple 1D Wasserstein distance without scipy (downsample to equal lengths)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    x = np.sort(x)
    y = np.sort(y)
    n = min(x.size, y.size)
    xi = x[np.linspace(0, x.size - 1, n).astype(int)]
    yi = y[np.linspace(0, y.size - 1, n).astype(int)]
    return float(np.mean(np.abs(xi - yi)))


def _histogram_prob(x: np.ndarray, bins: int = 50, range_: Optional[Tuple[float, float]] = None, eps: float = 1e-12):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return None, None
    hist, edges = np.histogram(x, bins=bins, range=range_)
    p = hist.astype(float) + eps
    p /= p.sum()
    return p, edges


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _distribution_divergences(x: np.ndarray, y: np.ndarray, bins: int = 50) -> Dict[str, Any]:
    """
    Histogram-based KL(A||B), KL(B||A), JS(A,B).
    Uses shared bin edges from combined data for stability.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return {"error": "empty distribution(s)"}

    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if lo == hi:
        return {"error": "degenerate distributions (all values identical)", "value": lo}

    p, _ = _histogram_prob(x, bins=bins, range_=(lo, hi))
    q, _ = _histogram_prob(y, bins=bins, range_=(lo, hi))
    assert p is not None and q is not None

    return {
        "bins": int(bins),
        "range": [lo, hi],
        "kl_x_to_y": _kl_divergence(p, q),
        "kl_y_to_x": _kl_divergence(q, p),
        "js": _js_divergence(p, q),
    }


def _parse_targets(args_targets: Optional[List[str]], args_target: Optional[str], df_cols: Sequence[str]) -> List[str]:
    targets: List[str] = []
    if args_target:
        targets.append(args_target)

    if args_targets:
        for item in args_targets:
            if not item:
                continue
            parts = [p.strip() for p in item.split(",") if p.strip()]
            targets.extend(parts)

    # env fallback
    if not targets:
        env_multi = os.environ.get("EDA_TARGET_COLUMNS", "").strip()
        env_single = os.environ.get("EDA_TARGET_COLUMN", "").strip()
        if env_multi:
            targets.extend([p.strip() for p in env_multi.split(",") if p.strip()])
        elif env_single:
            targets.append(env_single)

    targets = _dedupe_preserve_order([t for t in targets if t])

    # auto targets if still none
    if not targets:
        targets = [c for c in DEFAULT_TARGET_CANDIDATES if c in df_cols]

    return targets


def _sanitize_filename(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out[:120] if out else "col"


def _safe_json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return str(x)


def _empty_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    # Replace empty/whitespace-only strings with NaN across the dataframe.
    return df.replace(r"^\s*$", np.nan, regex=True)


def _looks_like_leading_zero_id(series: pd.Series, sample_n: int = 5000) -> bool:
    """
    Heuristic: if many non-null values look like integers with leading zeros (e.g., 00123),
    treat as ID-like and avoid classifying as numeric.
    """
    s = series.dropna().astype(str)
    if s.empty:
        return False
    if len(s) > sample_n:
        s = s.sample(sample_n, random_state=0)
    pat = re.compile(r"^0\d+$")
    frac = float(s.map(lambda x: bool(pat.match(x.strip()))).mean())
    return frac >= 0.2


# -------------------------
# RDKit helpers
# -------------------------
def _mols_from_smiles(smiles: pd.Series, max_mols: Optional[int], seed: int) -> Tuple[List[Any], List[int], Dict[str, Any]]:
    if not RDKIT_AVAILABLE:
        return [], [], {"available": False, "error": "RDKit not installed"}

    s = smiles.fillna("").astype(str)
    total = int(len(s))
    empty = int((s.str.strip() == "").sum())

    idx = np.arange(total, dtype=int)
    if max_mols is not None and total > max_mols:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_mols, replace=False)
        idx = np.sort(idx)

    invalid_examples: List[str] = []
    invalid_limit = 10

    mols: List[Any] = []
    valid_indices: List[int] = []
    valid = 0
    parse_fail = 0
    exceptions = 0

    for i in idx:
        smi = str(s.iloc[i]).strip()
        if not smi:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            exceptions += 1
            if len(invalid_examples) < invalid_limit:
                invalid_examples.append(smi)
            continue
        if mol is None:
            parse_fail += 1
            if len(invalid_examples) < invalid_limit:
                invalid_examples.append(smi)
            continue
        valid += 1
        mols.append(mol)
        valid_indices.append(int(i))

    stats = {
        "available": True,
        "total_smiles_rows_considered": int(len(idx)),
        "total_smiles_rows_original": total,
        "empty_or_whitespace_original": empty,
        "valid_mols": int(valid),
        "parse_failures": int(parse_fail),
        "rdkit_exceptions": int(exceptions),
        "invalid_smiles_examples": invalid_examples,
        "sampled": bool(max_mols is not None and total > max_mols),
        "max_mols": max_mols,
    }
    return mols, valid_indices, stats


def _rdkit_descriptor_panel() -> Dict[str, Any]:
    # A reasonably standard panel; used for stats + compare-to embeddings.
    return {
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
        "MolMR": Crippen.MolMR,
        "QED": Descriptors.qed,
    }


def _compute_descriptor_matrix(mols: List[Any], desc_funcs: Dict[str, Any]) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    cols = list(desc_funcs.keys())
    failures: Dict[str, int] = {k: 0 for k in cols}
    rows: List[List[float]] = []

    for m in mols:
        row: List[float] = []
        for name, fn in desc_funcs.items():
            try:
                row.append(float(fn(m)))
            except Exception:
                failures[name] += 1
                row.append(float("nan"))
        rows.append(row)

    X = np.asarray(rows, dtype=float) if rows else np.zeros((0, len(cols)), dtype=float)
    return X, cols, failures


def _rdkit_stats(mols: List[Any], fp_pair_samples: int, seed: int) -> Dict[str, Any]:
    if not RDKIT_AVAILABLE:
        return {"available": False}
    out: Dict[str, Any] = {"available": True}
    if not mols:
        out["error"] = "no valid mols parsed"
        return out

    desc_funcs = _rdkit_descriptor_panel()
    X, cols, desc_fail = _compute_descriptor_matrix(mols, desc_funcs)

    out["descriptor_failures"] = desc_fail
    out["descriptor_stats"] = {cols[i]: _summarize_numeric_array(X[:, i]) for i in range(len(cols))}

    # Scaffolds
    scaff = []
    for m in mols:
        try:
            scaff.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False))
        except Exception:
            scaff.append("")
    scaff_s = pd.Series(scaff, dtype="string")
    vc = scaff_s.value_counts(dropna=True)

    out["scaffolds"] = {
        "unique_scaffolds": int(scaff_s.nunique(dropna=True)),
        "top_scaffolds": [
            {"scaffold": str(idx), "count": int(cnt)}
            for idx, cnt in vc.head(20).items()
            if str(idx).strip() != ""
        ],
    }

    # InChIKey duplicates (if available)
    out["inchikey"] = {"available": bool(INCHI_AVAILABLE)}
    if INCHI_AVAILABLE:
        inchis: List[str] = []
        inchi_fail = 0
        for m in mols:
            try:
                inchis.append(str(MolToInchiKey(m)))
            except Exception:
                inchi_fail += 1
        inchi_s = pd.Series(inchis, dtype="string")
        out["inchikey"].update(
            {
                "computed": int(inchi_s.size),
                "failures": int(inchi_fail),
                "unique": int(inchi_s.nunique(dropna=True)),
                "duplicate_count": int(inchi_s.duplicated().sum()),
            }
        )

    # Fingerprint diversity (approx)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in mols]
    n = len(fps)
    rng = np.random.default_rng(seed)

    if n >= 2:
        pairs = min(fp_pair_samples, n * (n - 1) // 2)
        sims: List[float] = []
        for _ in range(pairs):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n - 1))
            if j >= i:
                j += 1
            sims.append(float(DataStructs.TanimotoSimilarity(fps[i], fps[j])))

        sims_arr = np.asarray(sims, dtype=float)
        out["fingerprint_diversity"] = {
            "pairs_sampled": int(pairs),
            "mean_tanimoto": float(np.mean(sims_arr)),
            "median_tanimoto": float(np.median(sims_arr)),
            "mean_distance_1_minus_tanimoto": float(np.mean(1.0 - sims_arr)),
        }
    else:
        out["fingerprint_diversity"] = {"error": "not enough mols for similarity"}

    return out


def _morgan_fp(mol: Any, radius: int = 2, nbits: int = 2048) -> Any:
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def _nn_max_sims(fps_query: List[Any], fps_ref: List[Any]) -> np.ndarray:
    sims = []
    for fp in fps_query:
        s = DataStructs.BulkTanimotoSimilarity(fp, fps_ref)
        sims.append(float(max(s)) if s else 0.0)
    return np.asarray(sims, dtype=float)


def _find_score_col(df: pd.DataFrame) -> Optional[str]:
    # Try case-insensitive match for "score"
    for c in df.columns:
        if c.lower() == "score":
            return c
    return None


# -------------------------
# Compare chemical space (RDKit + sklearn)
# -------------------------
def _compare_chemical_space(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    smiles_col_a: str,
    smiles_col_b: str,
    targets: Sequence[str],
    max_mols: int,
    seed: int,
    kl_bins: int,
    do_tsne: bool,
    tsne_sample: int,
    outdir: Optional[Path],
    do_umap: bool,
    umap_sample: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
) -> Dict[str, Any]:
    if not RDKIT_AVAILABLE:
        return {"available": False, "error": "RDKit not installed"}

    molA, idxA, statA = _mols_from_smiles(df_a[smiles_col_a], max_mols=max_mols, seed=seed)
    molB, idxB, statB = _mols_from_smiles(df_b[smiles_col_b], max_mols=max_mols, seed=seed + 1)

    out: Dict[str, Any] = {
        "available": True,
        "dataset_A": statA,
        "dataset_B": statB,
        "smiles_col_A": smiles_col_a,
        "smiles_col_B": smiles_col_b,
    }
    if not molA or not molB:
        out["error"] = "not enough valid mols in one or both datasets"
        return out

    # -------- Tanimoto max-sim distributions (ECFP4 == Morgan r=2)
    fpsA = [_morgan_fp(m) for m in molA]
    fpsB = [_morgan_fp(m) for m in molB]
    sims_A_to_B = _nn_max_sims(fpsA, fpsB)
    sims_B_to_A = _nn_max_sims(fpsB, fpsA)

    out["tanimoto_max_similarity"] = {
        "A_to_B": _summarize_numeric_array(sims_A_to_B),
        "B_to_A": _summarize_numeric_array(sims_B_to_A),
        "coverage": {
            "A_to_B": {str(t): float(np.mean(sims_A_to_B >= t)) for t in (0.6, 0.7, 0.8, 0.9)},
            "B_to_A": {str(t): float(np.mean(sims_B_to_A >= t)) for t in (0.6, 0.7, 0.8, 0.9)},
        },
        "hist": {
            "bins": 50,
            "A_to_B": np.histogram(sims_A_to_B, bins=50, range=(0.0, 1.0))[0].tolist(),
            "B_to_A": np.histogram(sims_B_to_A, bins=50, range=(0.0, 1.0))[0].tolist(),
        },
    }

    # -------- Descriptor matrix (shared panel used for shift + PCA/tSNE/UMAP)
    desc_funcs = _rdkit_descriptor_panel()
    XA, dcols, _ = _compute_descriptor_matrix(molA, desc_funcs)
    XB, _, _ = _compute_descriptor_matrix(molB, desc_funcs)

    # Extract scores if available
    score_col_A = _find_score_col(df_a)
    score_col_B = _find_score_col(df_b)
    scores_combined = None

    if score_col_A and score_col_B:
        try:
            sA = pd.to_numeric(df_a[score_col_A].iloc[idxA], errors="coerce").to_numpy(dtype=float)
            sB = pd.to_numeric(df_b[score_col_B].iloc[idxB], errors="coerce").to_numpy(dtype=float)
            scores_combined = np.concatenate([sA, sB])
        except Exception:
            scores_combined = None

    # Wasserstein per descriptor
    out["descriptor_shift"] = {
        "wasserstein_1d": {c: _wasserstein_1d(XA[:, i], XB[:, i]) for i, c in enumerate(dcols)}
    }

    # KL/JS per descriptor
    out["descriptor_divergence_kl_js"] = {
        c: _distribution_divergences(XA[:, i], XB[:, i], bins=kl_bins) for i, c in enumerate(dcols)
    }

    # -------- PCA 2D (descriptor panel)
    out["pca_descriptor_2d"] = {"available": False}
    if SKLEARN_AVAILABLE and (XA.shape[0] >= 2) and (XB.shape[0] >= 2):
        X = np.vstack([XA, XB])
        labels = np.array([0] * XA.shape[0] + [1] * XB.shape[0], dtype=int)

        # Replace NaNs with column means (PCA/t-SNE/UMAP can't handle NaNs)
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        Z = pca.fit_transform(Xs)
        ZA = Z[: XA.shape[0]]
        ZB = Z[XA.shape[0] :]

        pca_info: Dict[str, Any] = {
            "available": True,
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "A_centroid": [float(np.mean(ZA[:, 0])), float(np.mean(ZA[:, 1]))],
            "B_centroid": [float(np.mean(ZB[:, 0])), float(np.mean(ZB[:, 1]))],
        }

        try:
            if Z.shape[0] >= 10:
                pca_info["silhouette_score_in_PCA2"] = float(silhouette_score(Z, labels))
        except Exception as e:
            pca_info["silhouette_error"] = str(e)

        if outdir is not None:
            outdir.mkdir(parents=True, exist_ok=True)
            k = min(20000, Z.shape[0])
            rng = np.random.default_rng(seed)
            take = rng.choice(np.arange(Z.shape[0]), size=k, replace=False) if Z.shape[0] > k else np.arange(Z.shape[0])

            csv_data = {
                "pc1": Z[take, 0],
                "pc2": Z[take, 1],
                "dataset": np.where(labels[take] == 0, "A", "B"),
            }
            if scores_combined is not None:
                csv_data["score"] = scores_combined[take]

            df_coords = pd.DataFrame(csv_data)
            coords_path = outdir / "compare_pca2_coords.csv"
            df_coords.to_csv(coords_path, index=False)
            pca_info["coords_csv"] = str(coords_path)

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # Plot A vs B
                plt.figure()
                plt.scatter(df_coords.loc[df_coords["dataset"] == "A", "pc1"], df_coords.loc[df_coords["dataset"] == "A", "pc2"], s=6, alpha=0.5, label="A")
                plt.scatter(df_coords.loc[df_coords["dataset"] == "B", "pc1"], df_coords.loc[df_coords["dataset"] == "B", "pc2"], s=6, alpha=0.5, label="B")
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.legend()
                plt.tight_layout()
                plot_path = outdir / "compare_pca2.png"
                plt.savefig(plot_path, dpi=200)
                plt.close()
                pca_info["plot_png"] = str(plot_path)

                # Plot Score Heatmap
                if scores_combined is not None:
                    # Sort by score to plot high scores on top (usually more interesting)
                    df_s = df_coords.sort_values("score")
                    plt.figure()
                    sc = plt.scatter(df_s["pc1"], df_s["pc2"], c=df_s["score"], s=6, alpha=0.7, cmap="viridis")
                    plt.xlabel("PC1")
                    plt.ylabel("PC2")
                    plt.colorbar(sc, label="Score")
                    plt.tight_layout()
                    plot_path_s = outdir / "compare_pca2_score.png"
                    plt.savefig(plot_path_s, dpi=200)
                    plt.close()
                    pca_info["plot_score_png"] = str(plot_path_s)
            except Exception as e:
                pca_info["plot_error"] = str(e)

        out["pca_descriptor_2d"] = pca_info

    # -------- Optional t-SNE (descriptor panel)
    out["tsne_descriptor_2d"] = {"available": False}
    if do_tsne and SKLEARN_AVAILABLE:
        try:
            X = np.vstack([XA, XB])
            labels = np.array([0] * XA.shape[0] + [1] * XB.shape[0], dtype=int)
            n = X.shape[0]
            rng = np.random.default_rng(seed)

            if n > tsne_sample:
                take = rng.choice(np.arange(n), size=tsne_sample, replace=False)
                X_sub = X[take]
                y_sub = labels[take]
            else:
                take = np.arange(n)
                X_sub = X
                y_sub = labels

            # NaN -> column means
            col_means = np.nanmean(X_sub, axis=0)
            inds = np.where(np.isnan(X_sub))
            X_sub[inds] = np.take(col_means, inds[1])

            Xs = StandardScaler().fit_transform(X_sub)

            # Clamp perplexity
            n_used = Xs.shape[0]
            perplexity = min(30.0, max(5.0, float((n_used - 1) // 3)))
            tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed)
            Z = tsne.fit_transform(Xs)

            tsne_info: Dict[str, Any] = {"available": True, "n_used": int(Z.shape[0]), "perplexity": float(perplexity)}
            if outdir is not None:
                outdir.mkdir(parents=True, exist_ok=True)
                csv_data = {"tsne1": Z[:, 0], "tsne2": Z[:, 1], "dataset": np.where(y_sub == 0, "A", "B")}
                if scores_combined is not None:
                    csv_data["score"] = scores_combined[take]
                
                df_coords = pd.DataFrame(csv_data)
                coords_path = outdir / "compare_tsne2_coords.csv"
                df_coords.to_csv(coords_path, index=False)
                tsne_info["coords_csv"] = str(coords_path)

                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    # Plot A vs B
                    plt.figure()
                    plt.scatter(df_coords.loc[df_coords["dataset"] == "A", "tsne1"], df_coords.loc[df_coords["dataset"] == "A", "tsne2"], s=6, alpha=0.5, label="A")
                    plt.scatter(df_coords.loc[df_coords["dataset"] == "B", "tsne1"], df_coords.loc[df_coords["dataset"] == "B", "tsne2"], s=6, alpha=0.5, label="B")
                    plt.xlabel("t-SNE 1")
                    plt.ylabel("t-SNE 2")
                    plt.legend()
                    plt.tight_layout()
                    plot_path = outdir / "compare_tsne2.png"
                    plt.savefig(plot_path, dpi=200)
                    plt.close()
                    tsne_info["plot_png"] = str(plot_path)

                    # Plot Score Heatmap
                    if scores_combined is not None:
                        df_s = df_coords.sort_values("score")
                        plt.figure()
                        sc = plt.scatter(df_s["tsne1"], df_s["tsne2"], c=df_s["score"], s=6, alpha=0.7, cmap="viridis")
                        plt.xlabel("t-SNE 1")
                        plt.ylabel("t-SNE 2")
                        plt.colorbar(sc, label="Score")
                        plt.tight_layout()
                        plot_path_s = outdir / "compare_tsne2_score.png"
                        plt.savefig(plot_path_s, dpi=200)
                        plt.close()
                        tsne_info["plot_score_png"] = str(plot_path_s)
                except Exception as e:
                    tsne_info["plot_error"] = str(e)

            out["tsne_descriptor_2d"] = tsne_info
        except Exception as e:
            out["tsne_descriptor_2d"] = {"available": False, "error": str(e)}

    # -------- Optional UMAP 2D (descriptor panel)
    out["umap_descriptor_2d"] = {"available": False}
    if do_umap:
        if not UMAP_AVAILABLE:
            out["umap_descriptor_2d"] = {"available": False, "error": "umap-learn not installed"}
        elif not SKLEARN_AVAILABLE:
            out["umap_descriptor_2d"] = {"available": False, "error": "scikit-learn not available"}
        else:
            try:
                X = np.vstack([XA, XB])
                labels = np.array([0] * XA.shape[0] + [1] * XB.shape[0], dtype=int)
                n = X.shape[0]
                rng = np.random.default_rng(seed)

                if n > umap_sample:
                    take = rng.choice(np.arange(n), size=umap_sample, replace=False)
                    X_sub = X[take]
                    y_sub = labels[take]
                else:
                    take = np.arange(n)
                    X_sub = X
                    y_sub = labels

                # NaN -> column means
                col_means = np.nanmean(X_sub, axis=0)
                inds = np.where(np.isnan(X_sub))
                X_sub[inds] = np.take(col_means, inds[1])

                Xs = StandardScaler().fit_transform(X_sub)

                n_used = Xs.shape[0]
                n_neighbors = int(min(umap_n_neighbors, max(2, n_used - 1)))

                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=float(umap_min_dist),
                    metric=str(umap_metric),
                    random_state=seed,
                )
                Z = reducer.fit_transform(Xs)

                umap_info: Dict[str, Any] = {
                    "available": True,
                    "n_used": int(Z.shape[0]),
                    "params": {
                        "n_neighbors": int(n_neighbors),
                        "min_dist": float(umap_min_dist),
                        "metric": str(umap_metric),
                    },
                }

                if outdir is not None:
                    outdir.mkdir(parents=True, exist_ok=True)
                    csv_data = {"umap1": Z[:, 0], "umap2": Z[:, 1], "dataset": np.where(y_sub == 0, "A", "B")}
                    if scores_combined is not None:
                        csv_data["score"] = scores_combined[take]
                    
                    df_coords = pd.DataFrame(csv_data)
                    coords_path = outdir / "compare_umap2_coords.csv"
                    df_coords.to_csv(coords_path, index=False)
                    umap_info["coords_csv"] = str(coords_path)

                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt

                        # Plot A vs B
                        plt.figure()
                        plt.scatter(df_coords.loc[df_coords["dataset"] == "A", "umap1"], df_coords.loc[df_coords["dataset"] == "A", "umap2"], s=6, alpha=0.5, label="A")
                        plt.scatter(df_coords.loc[df_coords["dataset"] == "B", "umap1"], df_coords.loc[df_coords["dataset"] == "B", "umap2"], s=6, alpha=0.5, label="B")
                        plt.xlabel("UMAP 1")
                        plt.ylabel("UMAP 2")
                        plt.legend()
                        plt.tight_layout()
                        plot_path = outdir / "compare_umap2.png"
                        plt.savefig(plot_path, dpi=200)
                        plt.close()
                        umap_info["plot_png"] = str(plot_path)

                        # Plot Score Heatmap
                        if scores_combined is not None:
                            df_s = df_coords.sort_values("score")
                            plt.figure()
                            sc = plt.scatter(df_s["umap1"], df_s["umap2"], c=df_s["score"], s=6, alpha=0.7, cmap="viridis")
                            plt.xlabel("UMAP 1")
                            plt.ylabel("UMAP 2")
                            plt.colorbar(sc, label="Score")
                            plt.tight_layout()
                            plot_path_s = outdir / "compare_umap2_score.png"
                            plt.savefig(plot_path_s, dpi=200)
                            plt.close()
                            umap_info["plot_score_png"] = str(plot_path_s)
                    except Exception as e:
                        umap_info["plot_error"] = str(e)

                out["umap_descriptor_2d"] = umap_info
            except Exception as e:
                out["umap_descriptor_2d"] = {"available": False, "error": str(e)}

    # -------- Scaffold overlap
    scA = [MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in molA]
    scB = [MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in molB]
    setA, setB = set(scA), set(scB)
    out["scaffold_overlap"] = {
        "unique_A": int(len(setA)),
        "unique_B": int(len(setB)),
        "jaccard": float(len(setA & setB) / max(len(setA | setB), 1)),
    }

    # -------- Target/property divergences (KL/JS + Wasserstein)
    target_shift: Dict[str, Any] = {}
    for t in targets:
        if t in df_a.columns and t in df_b.columns:
            ya = pd.to_numeric(df_a[t], errors="coerce").to_numpy(dtype=float)
            yb = pd.to_numeric(df_b[t], errors="coerce").to_numpy(dtype=float)
            target_shift[t] = {
                "wasserstein_1d": _wasserstein_1d(ya, yb),
                "kl_js": _distribution_divergences(ya, yb, bins=kl_bins),
                "A_summary": _summarize_numeric_array(ya),
                "B_summary": _summarize_numeric_array(yb),
            }
    out["target_shift"] = target_shift

    # Optional plots: tanimoto hist + target hists
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure()
            plt.hist(sims_A_to_B, bins=50, range=(0, 1), alpha=0.5, label="A→B")
            plt.hist(sims_B_to_A, bins=50, range=(0, 1), alpha=0.5, label="B→A")
            plt.xlabel("max Tanimoto similarity")
            plt.ylabel("count")
            plt.legend()
            plt.tight_layout()
            p = outdir / "compare_tanimoto_maxsim_hist.png"
            plt.savefig(p, dpi=200)
            plt.close()
            out["tanimoto_max_similarity"]["hist_plot_png"] = str(p)
        except Exception as e:
            out["tanimoto_max_similarity"]["hist_plot_error"] = str(e)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for t in targets:
                if t not in df_a.columns or t not in df_b.columns:
                    continue
                ya = pd.to_numeric(df_a[t], errors="coerce").dropna().to_numpy(dtype=float)
                yb = pd.to_numeric(df_b[t], errors="coerce").dropna().to_numpy(dtype=float)
                if ya.size == 0 or yb.size == 0:
                    continue

                lo = float(min(np.min(ya), np.min(yb)))
                hi = float(max(np.max(ya), np.max(yb)))
                if lo == hi:
                    continue

                plt.figure()
                plt.hist(ya, bins=60, range=(lo, hi), alpha=0.5, label="A")
                plt.hist(yb, bins=60, range=(lo, hi), alpha=0.5, label="B")
                plt.xlabel(t)
                plt.ylabel("count")
                plt.legend()
                plt.tight_layout()
                p = outdir / f"compare_hist_{_sanitize_filename(t)}.png"
                plt.savefig(p, dpi=200)
                plt.close()
                target_shift.setdefault(t, {})["hist_plot_png"] = str(p)
        except Exception as e:
            out.setdefault("plot_errors", []).append(str(e))

    return out


def _plot_single_dataset_embeddings(
    df: pd.DataFrame, mols: List[Any], valid_idx: List[int], cfg: EDAConfig, results: Dict[str, Any]
):
    if not SKLEARN_AVAILABLE:
        return

    outdir = cfg.outdir
    if not outdir:
        return

    outdir.mkdir(parents=True, exist_ok=True)

    # Compute descriptors
    desc_funcs = _rdkit_descriptor_panel()
    X, dcols, _ = _compute_descriptor_matrix(mols, desc_funcs)
    if X.shape[0] < 2:
        return

    # Handle NaNs in X
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Get scores
    score_col = _find_score_col(df)
    scores = None
    if score_col:
        try:
            # valid_idx corresponds to mols 1-to-1
            # df indices might not be monotonic 0..N, so use iloc with valid_idx
            scores = pd.to_numeric(df[score_col].iloc[valid_idx], errors="coerce").to_numpy(dtype=float)
        except Exception:
            scores = None

    # Helper for plotting
    def _do_plot(coords, name_suffix, label_x, label_y, used_scores):
        df_c = pd.DataFrame(coords, columns=["x", "y"])
        if used_scores is not None:
            df_c["score"] = used_scores

        # Save coords
        csv_path = outdir / f"single_{name_suffix}_coords.csv"
        df_c.to_csv(csv_path, index=False)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # 1. Plain plot (blue dots)
            plt.figure()
            plt.scatter(df_c["x"], df_c["y"], s=6, alpha=0.5)
            plt.xlabel(label_x)
            plt.ylabel(label_y)
            plt.tight_layout()
            plt.savefig(outdir / f"single_{name_suffix}.png", dpi=200)
            plt.close()

            # 2. Score plot
            if used_scores is not None:
                df_s = df_c.sort_values("score")
                plt.figure()
                sc = plt.scatter(df_s["x"], df_s["y"], c=df_s["score"], s=6, alpha=0.7, cmap="viridis")
                plt.colorbar(sc, label=score_col)
                plt.xlabel(label_x)
                plt.ylabel(label_y)
                plt.tight_layout()
                plt.savefig(outdir / f"single_{name_suffix}_score.png", dpi=200)
                plt.close()
        except Exception as e:
            print(f"Plot error ({name_suffix}): {e}")

    # PCA
    try:
        pca = PCA(n_components=2)
        Z_pca = pca.fit_transform(Xs)
        _do_plot(Z_pca, "pca2", "PC1", "PC2", scores)
        # Add stats to results
        if "pca_descriptor_2d_single" not in results["sklearn"]:
            results["sklearn"]["pca_descriptor_2d_single"] = {
                "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_]
            }
    except Exception as e:
        print(f"PCA error: {e}")

    # t-SNE
    if cfg.do_tsne:
        try:
            # Downsample if needed
            n = Xs.shape[0]
            if n > cfg.tsne_sample:
                rng = np.random.default_rng(cfg.seed)
                take = rng.choice(np.arange(n), size=cfg.tsne_sample, replace=False)
                Xs_sub = Xs[take]
                scores_sub = scores[take] if scores is not None else None
                _do_plot(
                    TSNE(
                        n_components=2, init="pca", learning_rate="auto", perplexity=min(30.0, max(5.0, float((Xs_sub.shape[0] - 1) // 3))), random_state=cfg.seed
                    ).fit_transform(Xs_sub),
                    "tsne2",
                    "t-SNE 1",
                    "t-SNE 2",
                    scores_sub,
                )
            else:
                perplexity = min(30.0, max(5.0, float((Xs.shape[0] - 1) // 3)))
                tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=cfg.seed)
                Z_tsne = tsne.fit_transform(Xs)
                _do_plot(Z_tsne, "tsne2", "t-SNE 1", "t-SNE 2", scores)
        except Exception as e:
            print(f"t-SNE error: {e}")

    # UMAP
    if cfg.do_umap and UMAP_AVAILABLE:
        try:
            # Downsample if needed
            n = Xs.shape[0]
            if n > cfg.umap_sample:
                rng = np.random.default_rng(cfg.seed)
                take = rng.choice(np.arange(n), size=cfg.umap_sample, replace=False)
                Xs_sub = Xs[take]
                scores_sub = scores[take] if scores is not None else None
                _do_plot(
                    umap.UMAP(
                        n_components=2,
                        n_neighbors=int(min(cfg.umap_n_neighbors, max(2, Xs_sub.shape[0] - 1))),
                        min_dist=float(cfg.umap_min_dist),
                        metric=str(cfg.umap_metric),
                        random_state=cfg.seed,
                    ).fit_transform(Xs_sub),
                    "umap2",
                    "UMAP 1",
                    "UMAP 2",
                    scores_sub,
                )
            else:
                n_neighbors = int(min(cfg.umap_n_neighbors, max(2, Xs.shape[0] - 1)))
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=float(cfg.umap_min_dist),
                    metric=str(cfg.umap_metric),
                    random_state=cfg.seed,
                )
                Z_umap = reducer.fit_transform(Xs)
                _do_plot(Z_umap, "umap2", "UMAP 1", "UMAP 2", scores)
        except Exception as e:
            print(f"UMAP error: {e}")


# -------------------------
# Main EDA
# -------------------------
@dataclass(frozen=True)
class EDAConfig:
    top_n: int = 5
    corr_threshold: float = 0.9
    smiles_column: Optional[str] = None
    targets: Tuple[str, ...] = ()
    outdir: Optional[Path] = None

    # Data cleaning / numeric detection
    empty_as_missing: bool = True
    numeric_ratio_min: float = 0.95
    numeric_unique_min: int = 10
    corr_min_count: int = 50
    max_corr_cols: int = 200
    include_corr_matrices: bool = True
    max_high_pairs: int = 10000

    # RDKit sampling
    max_rdkit_mols: Optional[int] = 50000
    fp_pair_samples: int = 20000
    seed: int = 0

    # Compare metrics knobs
    kl_bins: int = 50
    do_tsne: bool = False
    tsne_sample: int = 5000

    do_umap: bool = False
    umap_sample: int = 20000
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"

    # Output bounds
    max_firstcol_dup_examples: int = 200


def analyze_csv(input_file: str, output_file: str, cfg: EDAConfig, compare_to: Optional[Sequence[str]]) -> None:
    df = pd.read_csv(input_file, dtype=str, low_memory=False)
    if cfg.empty_as_missing:
        df = _empty_to_nan(df)

    total_rows = int(len(df))
    total_cols = int(len(df.columns))
    total_cells = int(total_rows * total_cols)
    missing_cells = int(df.isna().sum().sum())

    smiles_col = cfg.smiles_column if cfg.smiles_column else (df.columns[0] if total_cols else None)
    if smiles_col is None or smiles_col not in df.columns:
        smiles_col = df.columns[0] if total_cols else None

    targets = list(cfg.targets) if cfg.targets else _parse_targets(None, None, df.columns.tolist())

    results: Dict[str, Any] = {
        "headers": df.columns.tolist(),
        "metadata": {
            "total_rows": total_rows,
            "columns": total_cols,
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percent": round((missing_cells / total_cells * 100) if total_cells else 0.0, 6),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_bytes": int(df.memory_usage(deep=True).sum()),
        },
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": getattr(pd, "__version__", None),
            "sklearn": (getattr(__import__("sklearn"), "__version__", None) if SKLEARN_AVAILABLE else None),
            "rdkit": (getattr(__import__("rdkit"), "__version__", None) if RDKIT_AVAILABLE else None),
            "umap": (getattr(umap, "__version__", None) if UMAP_AVAILABLE else None),
        },
        "smiles_column": smiles_col,
        "targets": targets,
        "config": {
            "empty_as_missing": cfg.empty_as_missing,
            "numeric_ratio_min": cfg.numeric_ratio_min,
            "numeric_unique_min": cfg.numeric_unique_min,
            "corr_min_count": cfg.corr_min_count,
            "max_corr_cols": cfg.max_corr_cols,
            "include_corr_matrices": cfg.include_corr_matrices,
            "max_high_pairs": cfg.max_high_pairs,
        },
        "column_missing": {},
        "column_unique": {},
        "column_mode": {},
        "column_top_values": {},
        "column_string_length": {},
        "column_entropy": {},
        "column_numeric_count": {},
        "column_numeric_meta": {},
        "column_stats": {},
        "first_column_duplicates": {},
        "correlations": {},
        "sklearn": {},
        "rdkit": {},
        "compare_to": {},
        "artifacts": {},
    }

    # Duplicates in first column (bounded)
    if total_cols > 0:
        first_col = df.iloc[:, 0]
        dups = first_col[first_col.duplicated()].dropna().astype(str)
        uniq_dups = pd.Series(dups.unique(), dtype="string")
        examples = uniq_dups.head(cfg.max_firstcol_dup_examples).tolist()
        results["first_column_duplicates"] = {
            "count_unique": int(len(uniq_dups)),
            "examples": examples,
            "examples_truncated": bool(len(uniq_dups) > len(examples)),
        }

    # Per-column stats and numeric extraction
    numeric_columns: List[str] = []
    numeric_data: Dict[str, pd.Series] = {}
    numeric_counts: Dict[str, int] = {}

    for col in df.columns:
        col_series = df[col]
        non_null = col_series.dropna()

        missing_count = int(total_rows - len(non_null))
        results["column_missing"][col] = {
            "missing": missing_count,
            "missing_percent": round((missing_count / total_rows * 100) if total_rows else 0.0, 6),
        }

        unique_count = int(non_null.nunique(dropna=True))
        results["column_unique"][col] = {
            "unique": unique_count,
            "unique_percent": round((unique_count / total_rows * 100) if total_rows else 0.0, 6),
        }

        # Mode/top-k (could be heavy for huge columns, but bounded by top_n output)
        if not non_null.empty:
            value_counts = non_null.value_counts(dropna=True)
            mode_value = value_counts.index[0]
            mode_count = int(value_counts.iloc[0])
            results["column_mode"][col] = {
                "mode": str(mode_value),
                "count": mode_count,
                "percent": round((mode_count / total_rows * 100) if total_rows else 0.0, 6),
            }
            top_vals = []
            for value, count in value_counts.head(cfg.top_n).items():
                top_vals.append(
                    {
                        "value": str(value),
                        "count": int(count),
                        "percent": round((int(count) / total_rows * 100) if total_rows else 0.0, 6),
                    }
                )
            results["column_top_values"][col] = top_vals
        else:
            results["column_mode"][col] = {"mode": None, "count": 0, "percent": 0.0}
            results["column_top_values"][col] = []

        # String lengths (if empty_as_missing=True, blanks largely disappear)
        if not non_null.empty:
            str_vals = non_null.astype(str)
            lengths = str_vals.str.len()
            blank_count = int((str_vals.str.strip() == "").sum())
            results["column_string_length"][col] = {
                "min": int(lengths.min()),
                "max": int(lengths.max()),
                "mean": round(float(lengths.mean()), 6),
                "median": round(float(lengths.median()), 6),
                "blank_count": blank_count,
            }
        else:
            results["column_string_length"][col] = {"min": None, "max": None, "mean": None, "median": None, "blank_count": 0}

        # Entropy (categorical-ish)
        results["column_entropy"][col] = round(_entropy_from_series(non_null), 8)

        # Numeric conversion + robust "is numeric" decision
        numeric_series = pd.to_numeric(col_series, errors="coerce")
        valid_numbers = numeric_series.dropna()
        n_non_null = int(len(non_null))
        n_numeric = int(valid_numbers.count())
        numeric_ratio = float(n_numeric / max(n_non_null, 1))

        results["column_numeric_count"][col] = int(n_numeric)

        unique_numeric = int(valid_numbers.nunique(dropna=True))
        is_id_like = ("id" in col.lower()) or _looks_like_leading_zero_id(col_series)

        is_numeric = (
            (n_non_null > 0)
            and (numeric_ratio >= cfg.numeric_ratio_min)
            and (unique_numeric >= cfg.numeric_unique_min)
            and (not is_id_like)
        )

        results["column_numeric_meta"][col] = {
            "numeric_ratio_among_non_null": round(numeric_ratio, 6),
            "unique_numeric": int(unique_numeric),
            "id_like": bool(is_id_like),
            "classified_as_numeric": bool(is_numeric),
        }

        if is_numeric:
            numeric_columns.append(col)
            numeric_data[col] = numeric_series
            numeric_counts[col] = n_numeric

            count = int(n_numeric)
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
                "mean": round(mean_val, 10),
                "median": round(float(valid_numbers.median()), 10),
                "std": round(std_val, 12),
                "variance": round(var_val, 12),
                "q1": round(q1, 12),
                "q3": round(q3, 12),
                "iqr": round(iqr, 12),
                "p5": round(p5, 12),
                "p95": round(p95, 12),
                "skew": round(float(valid_numbers.skew()), 12),
                "kurtosis": round(float(valid_numbers.kurt()), 12),
                "zeros": zeros,
                "negatives": negatives,
                "outliers_iqr": outliers_iqr,
                "outliers_zscore": outliers_z,
                "count": count,
                "count_percent": round((count / total_rows * 100) if total_rows else 0.0, 6),
            }

    # Correlations (guarded + bounded)
    if numeric_columns:
        # Keep only columns with enough observed numeric values
        corr_candidates = [c for c in numeric_columns if numeric_counts.get(c, 0) >= cfg.corr_min_count]

        # If too many columns, choose by variance (descending) to keep correlations tractable.
        if len(corr_candidates) > cfg.max_corr_cols:
            var_list: List[Tuple[str, float]] = []
            for c in corr_candidates:
                var = float(results["column_stats"].get(c, {}).get("variance", 0.0))
                var_list.append((c, var))
            var_list.sort(key=lambda x: x[1], reverse=True)
            selected = [c for c, _ in var_list[: cfg.max_corr_cols]]
            dropped = [c for c, _ in var_list[cfg.max_corr_cols :]]
            corr_cols = selected
            results["correlations"]["column_selection"] = {
                "selected_by": "variance_desc",
                "selected_n": int(len(selected)),
                "dropped_n": int(len(dropped)),
                "max_corr_cols": int(cfg.max_corr_cols),
            }
        else:
            corr_cols = corr_candidates

        if len(corr_cols) >= 2:
            numeric_df = pd.DataFrame({c: numeric_data[c] for c in corr_cols})

            pearson_corr = numeric_df.corr(method="pearson")
            spearman_corr = numeric_df.corr(method="spearman")

            results["correlations"]["n_columns_used"] = int(len(corr_cols))
            results["correlations"]["columns_used"] = corr_cols

            if cfg.include_corr_matrices:
                results["correlations"]["pearson"] = pearson_corr.round(8).to_dict()
                results["correlations"]["spearman"] = spearman_corr.round(8).to_dict()

            # High-correlation pairs (bounded)
            def _high_pairs(corr: pd.DataFrame, label: str) -> List[Dict[str, Any]]:
                cols = corr.columns.tolist()
                mat = corr.to_numpy()
                pairs: List[Dict[str, Any]] = []
                n = len(cols)

                # upper triangle scan with early cutoff
                for i in range(n):
                    for j in range(i + 1, n):
                        v = mat[i, j]
                        if np.isnan(v):
                            continue
                        if abs(v) >= cfg.corr_threshold:
                            pairs.append({"col_a": cols[i], "col_b": cols[j], "corr": round(float(v), 8)})
                            if len(pairs) >= cfg.max_high_pairs:
                                results["correlations"][f"high_pairs_{label}_truncated"] = True
                                return pairs
                results["correlations"][f"high_pairs_{label}_truncated"] = False
                return pairs

            results["correlations"]["high_pairs_pearson"] = _high_pairs(pearson_corr, "pearson")
            results["correlations"]["high_pairs_spearman"] = _high_pairs(spearman_corr, "spearman")

    # RDKit analysis
    results["rdkit"]["available"] = bool(RDKIT_AVAILABLE)
    if RDKIT_AVAILABLE and smiles_col is not None and smiles_col in df.columns:
        mols, valid_indices, smi_stats = _mols_from_smiles(df[smiles_col], max_mols=cfg.max_rdkit_mols, seed=cfg.seed)
        results["rdkit"]["smiles_parse"] = smi_stats
        results["rdkit"].update(_rdkit_stats(mols, fp_pair_samples=cfg.fp_pair_samples, seed=cfg.seed))

        # Single dataset embeddings if outdir is set
        if cfg.outdir is not None and mols:
             _plot_single_dataset_embeddings(
                df, mols, valid_indices, cfg, results
             )
    else:
        results["rdkit"]["error"] = "RDKit not installed or SMILES column missing"

    # sklearn analysis
    results["sklearn"]["available"] = bool(SKLEARN_AVAILABLE)
    if SKLEARN_AVAILABLE and numeric_columns and total_rows > 0:
        numeric_df = pd.DataFrame(numeric_data)
        X = numeric_df.copy()

        # Fill NaNs per-column with mean; remaining NaNs -> 0
        for col in X.columns:
            mu = X[col].mean()
            X[col] = X[col].fillna(mu)
        X = X.fillna(0.0)

        # VarianceThreshold
        variance_threshold = 1e-12
        try:
            vt = VarianceThreshold(threshold=variance_threshold)
            vt.fit(X)
            support = vt.get_support()
            kept_cols = list(np.array(X.columns)[support])
            removed_cols = [c for c in X.columns if c not in kept_cols]
            results["sklearn"]["variance_threshold"] = {
                "threshold": variance_threshold,
                "kept_columns": kept_cols,
                "removed_columns": removed_cols,
            }
        except Exception as e:
            results["sklearn"]["variance_threshold_error"] = str(e)

        # PCA explained variance
        try:
            if X.shape[1] >= 1 and X.shape[0] >= 2:
                n_components = int(min(10, X.shape[1], X.shape[0]))
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                pca = PCA(n_components=n_components)
                pca.fit(X_scaled)
                results["sklearn"]["pca"] = {
                    "n_components": n_components,
                    "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
                    "explained_variance_ratio_cumulative": [float(v) for v in np.cumsum(pca.explained_variance_ratio_)],
                }
        except Exception as e:
            results["sklearn"]["pca_error"] = str(e)

        # Mutual information for each target (guard high-cardinality classification)
        results["sklearn"]["mutual_information"] = {}
        for tcol in targets:
            if tcol not in df.columns:
                results["sklearn"]["mutual_information"][tcol] = {"error": "target column not found"}
                continue

            X_feat = X.drop(columns=[tcol], errors="ignore")
            if X_feat.shape[1] < 1:
                results["sklearn"]["mutual_information"][tcol] = {"error": "no numeric feature columns available"}
                continue

            target_series = df[tcol]
            if cfg.empty_as_missing:
                target_series = target_series.replace(r"^\s*$", np.nan, regex=True)

            target_numeric = pd.to_numeric(target_series, errors="coerce")
            numeric_ratio = float(target_numeric.notna().mean()) if total_rows else 0.0
            nunique = int(target_series.dropna().nunique(dropna=True))

            try:
                if numeric_ratio > 0.9 and int(target_numeric.nunique(dropna=True)) > 20:
                    y = target_numeric.fillna(target_numeric.mean())
                    mi = mutual_info_regression(X_feat, y)
                    mi_type = "regression"
                else:
                    # Skip MI for very high-cardinality categorical targets by default
                    if nunique > 200:
                        results["sklearn"]["mutual_information"][tcol] = {
                            "target_column": tcol,
                            "type": "classification",
                            "skipped": True,
                            "reason": f"high-cardinality target (n_unique={nunique})",
                        }
                        continue
                    y = target_series.fillna("").astype(str)
                    y_codes, _ = pd.factorize(y)
                    mi = mutual_info_classif(X_feat, y_codes, discrete_features=False)
                    mi_type = "classification"

                mi_scores = sorted(
                    [{"column": col, "mi": float(score)} for col, score in zip(X_feat.columns, mi)],
                    key=lambda item: item["mi"],
                    reverse=True,
                )
                results["sklearn"]["mutual_information"][tcol] = {
                    "target_column": tcol,
                    "type": mi_type,
                    "scores_top_100": mi_scores[:100],
                }
            except Exception as e:
                results["sklearn"]["mutual_information"][tcol] = {"target_column": tcol, "error": str(e)}
    else:
        results["sklearn"]["error"] = "scikit-learn not installed or no numeric columns"

    # Compare-to mode
        # Compare-to mode (supports 1 or many compare files)
    if compare_to:
        # Backward-compatible behavior: if only one file is passed, keep the original JSON shape
        compare_list = list(compare_to)

        max_mols = int(cfg.max_rdkit_mols) if cfg.max_rdkit_mols is not None else 50000

        def _unique_key(existing: set[str], base: str) -> str:
            if base not in existing:
                existing.add(base)
                return base
            k = 2
            while f"{base}__{k}" in existing:
                k += 1
            new_base = f"{base}__{k}"
            existing.add(new_base)
            return new_base

        if len(compare_list) == 1:
            compare_path = compare_list[0]
            try:
                df2 = pd.read_csv(compare_path, dtype=str, low_memory=False)
                if cfg.empty_as_missing:
                    df2 = _empty_to_nan(df2)

                smiles_col_b = (
                    smiles_col if (smiles_col and smiles_col in df2.columns)
                    else (df2.columns[0] if len(df2.columns) else "")
                )

                # Preserve original artifact behavior for single compare: use cfg.outdir directly
                results["compare_to"] = _compare_chemical_space(
                    df_a=df,
                    df_b=df2,
                    smiles_col_a=str(smiles_col),
                    smiles_col_b=str(smiles_col_b),
                    targets=targets,
                    max_mols=max_mols,
                    seed=int(cfg.seed),
                    kl_bins=int(cfg.kl_bins),
                    do_tsne=bool(cfg.do_tsne),
                    tsne_sample=int(cfg.tsne_sample),
                    outdir=cfg.outdir,
                    do_umap=bool(cfg.do_umap),
                    umap_sample=int(cfg.umap_sample),
                    umap_n_neighbors=int(cfg.umap_n_neighbors),
                    umap_min_dist=float(cfg.umap_min_dist),
                    umap_metric=str(cfg.umap_metric),
                )
            except Exception as e:
                results["compare_to"] = {"available": False, "error": str(e)}
        else:
            # Multi-compare mode: store a map of comparisons, and prevent plot/CSV overwrites
            results["compare_to"] = {
                "available": True,
                "mode": "multi",
                "compare_files": [str(x) for x in compare_list],
                "comparisons": {},
            }

            used_keys: set[str] = set()
            for compare_path in compare_list:
                base_key = Path(compare_path).name
                key = _unique_key(used_keys, base_key)

                try:
                    df2 = pd.read_csv(compare_path, dtype=str, low_memory=False)
                    if cfg.empty_as_missing:
                        df2 = _empty_to_nan(df2)

                    smiles_col_b = (
                        smiles_col if (smiles_col and smiles_col in df2.columns)
                        else (df2.columns[0] if len(df2.columns) else "")
                    )

                    # Use per-compare subdir to avoid overwriting artifacts
                    compare_outdir = None
                    if cfg.outdir is not None:
                        tag = _sanitize_filename(Path(compare_path).stem)
                        compare_outdir = cfg.outdir / f"compare_to_{tag}"

                    results["compare_to"]["comparisons"][key] = _compare_chemical_space(
                        df_a=df,
                        df_b=df2,
                        smiles_col_a=str(smiles_col),
                        smiles_col_b=str(smiles_col_b),
                        targets=targets,
                        max_mols=max_mols,
                        seed=int(cfg.seed),
                        kl_bins=int(cfg.kl_bins),
                        do_tsne=bool(cfg.do_tsne),
                        tsne_sample=int(cfg.tsne_sample),
                        outdir=compare_outdir,
                        do_umap=bool(cfg.do_umap),
                        umap_sample=int(cfg.umap_sample),
                        umap_n_neighbors=int(cfg.umap_n_neighbors),
                        umap_min_dist=float(cfg.umap_min_dist),
                        umap_metric=str(cfg.umap_metric),
                    )
                except Exception as e:
                    results["compare_to"]["comparisons"][key] = {"available": False, "error": str(e)}


    # Optional artifacts
    if cfg.outdir is not None:
        outdir = cfg.outdir
        outdir.mkdir(parents=True, exist_ok=True)
        results["artifacts"]["outdir"] = str(outdir.resolve())

        # numeric column stats table
        if results["column_stats"]:
            stats_df = pd.DataFrame(results["column_stats"]).T
            stats_path = outdir / "numeric_column_stats.csv"
            stats_df.to_csv(stats_path, index=True)
            results["artifacts"]["numeric_column_stats_csv"] = str(stats_path)

        # top scaffolds table
        try:
            top_scaff = results.get("rdkit", {}).get("scaffolds", {}).get("top_scaffolds", [])
            if top_scaff:
                sc_df = pd.DataFrame(top_scaff)
                sc_path = outdir / "top_scaffolds.csv"
                sc_df.to_csv(sc_path, index=False)
                results["artifacts"]["top_scaffolds_csv"] = str(sc_path)
        except Exception:
            pass

        # per-target histograms (single dataset)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for t in targets:
                if t in df.columns:
                    y = pd.to_numeric(df[t], errors="coerce").dropna()
                    if y.empty:
                        continue
                    plt.figure()
                    plt.hist(y.values, bins=60)
                    plt.xlabel(t)
                    plt.ylabel("count")
                    plt.tight_layout()
                    p = outdir / f"hist_{_sanitize_filename(t)}.png"
                    plt.savefig(p, dpi=200)
                    plt.close()
        except Exception as e:
            results["artifacts"]["plot_error"] = str(e)

    # Save JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_safe_json_default)

    print(f"[OK] EDA saved: {output_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", help="Input CSV path")
    p.add_argument("output_json", nargs="?", default=None, help="Output JSON path (default: eda_<input>.json)")

    p.add_argument("--smiles-column", default="smiles", help="Column name to treat as SMILES (default: first column)")
    p.add_argument("--target", default="mp", help="Single target column (can also use --targets)")
    p.add_argument(
        "--targets",
        action="append",
        default=None,
        help='Target column(s). Repeatable or comma-separated. Example: --targets "Score,LogD (Lipo) (raw)"',
    )

    p.add_argument("--top-n", type=int, default=5, help="Top N values to report per column")
    p.add_argument("--corr-threshold", type=float, default=0.9, help="Correlation threshold for high-pairs")

    p.add_argument("--outdir", default=None, help="If set, save helper CSVs/plots to this directory")
    p.add_argument("--max-rdkit-mols", type=int, default=50000, help="Max SMILES rows to parse with RDKit (sampling)")
    p.add_argument("--fp-pair-samples", type=int, default=20000, help="Random fp pairs for diversity estimate")
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    p.add_argument(
    "--compare-to",
    nargs="+",
    default=None,
    help="Optional CSV(s) to compare chemical space against. Example: --compare-to B.csv C.csv D.csv",
)


    # Cleaning / numeric detection knobs
    p.add_argument("--no-empty-as-missing", action="store_true", help="Do not convert empty/whitespace strings to NaN")
    p.add_argument("--numeric-ratio-min", type=float, default=0.95, help="Min numeric parse ratio among non-null to classify numeric")
    p.add_argument("--numeric-unique-min", type=int, default=10, help="Min unique numeric values to classify numeric")
    p.add_argument("--corr-min-count", type=int, default=50, help="Min numeric count required to include column in correlations")
    p.add_argument("--max-corr-cols", type=int, default=200, help="Max numeric columns used for correlation matrices/pairs")
    p.add_argument("--no-corr-matrices", action="store_true", help="Do not store full correlation matrices in JSON (pairs only)")
    p.add_argument("--max-high-pairs", type=int, default=10000, help="Max high-correlation pairs stored (per corr type)")

    # compare metrics knobs (RDKit+sklearn)
    p.add_argument("--kl-bins", type=int, default=50, help="Bins for histogram-based KL/JS (default 50)")
    p.add_argument("--tsne", action="store_true", help="Also run t-SNE 2D on RDKit descriptor panel (compare-to only)")
    p.add_argument("--tsne-sample", type=int, default=5000, help="Max points for t-SNE (default 5000)")

    p.add_argument("--umap", action="store_true", help="Run UMAP 2D on RDKit descriptor panel (compare-to only)")
    p.add_argument("--umap-sample", type=int, default=20000, help="Max points for UMAP (default 20000)")
    p.add_argument("--umap-n-neighbors", type=int, default=15, help="UMAP n_neighbors (default 15)")
    p.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist (default 0.1)")
    p.add_argument("--umap-metric", type=str, default="euclidean", help="UMAP metric (default euclidean)")

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.exists(args.input_csv):
        print(f"File {args.input_csv} not found.")
        return 2

    if args.output_json:
        output_json = args.output_json
    else:
        base = os.path.splitext(os.path.basename(args.input_csv))[0]
        output_json = f"eda_{base}.json"

    # Preview cols for auto target selection
    df_preview = pd.read_csv(args.input_csv, dtype=str, nrows=5, low_memory=False)
    if not args.no_empty_as_missing:
        df_preview = _empty_to_nan(df_preview)
    targets = _parse_targets(args.targets, args.target, df_preview.columns.tolist())

    outdir = Path(args.outdir).expanduser() if args.outdir else None

    cfg = EDAConfig(
        top_n=int(args.top_n),
        corr_threshold=float(args.corr_threshold),
        smiles_column=args.smiles_column,
        targets=tuple(targets),
        outdir=outdir,
        empty_as_missing=not bool(args.no_empty_as_missing),
        numeric_ratio_min=float(args.numeric_ratio_min),
        numeric_unique_min=int(args.numeric_unique_min),
        corr_min_count=int(args.corr_min_count),
        max_corr_cols=int(args.max_corr_cols),
        include_corr_matrices=not bool(args.no_corr_matrices),
        max_high_pairs=int(args.max_high_pairs),
        max_rdkit_mols=int(args.max_rdkit_mols) if args.max_rdkit_mols else None,
        fp_pair_samples=int(args.fp_pair_samples),
        seed=int(args.seed),
        kl_bins=int(args.kl_bins),
        do_tsne=bool(args.tsne),
        tsne_sample=int(args.tsne_sample),
        do_umap=bool(args.umap),
        umap_sample=int(args.umap_sample),
        umap_n_neighbors=int(args.umap_n_neighbors),
        umap_min_dist=float(args.umap_min_dist),
        umap_metric=str(args.umap_metric),
    )

    analyze_csv(args.input_csv, output_json, cfg=cfg, compare_to=args.compare_to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
