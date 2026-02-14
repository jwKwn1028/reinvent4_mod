#!/usr/bin/env python3
"""
dedup_smiles_inchi_large.py

Deduplicate a large CSV (e.g., ~2M rows) by **chemical identity** using RDKit InChIKey.
This version **does NOT aggregate** any property columns (e.g., mp). It simply keeps
the first (or last) occurrence per InChIKey.

Laptop-friendly:
- Streams the CSV in chunks (does not load full file)
- Two backends:
  - python: fastest, keeps a Python set of seen InChIKeys (RAM-heavy for ~2M uniques)
  - sqlite: low-RAM, stores seen keys in an on-disk SQLite DB (slower but safer)

Behavior:
- Drops columns named "Unnamed:*"
- Parses SMILES -> Mol -> InChIKey
- Optionally replaces SMILES with canonical SMILES
- Keeps first or last row per InChIKey (no numeric aggregation)
- Optional audit CSV of removed rows (invalid + duplicates)

Examples
  # Keep FIRST occurrence (default), low-RAM backend
  python dedup_smiles_inchi_large.py --input mp.csv --out mp_dedup.csv --backend sqlite

  # Keep FIRST occurrence, replace SMILES with canonical SMILES
  python dedup_smiles_inchi_large.py --input mp.csv --out mp_dedup.csv --smiles-rep canonical --backend sqlite

  # Also write removed rows audit
  python dedup_smiles_inchi_large.py --input mp.csv --out mp_dedup.csv --removed mp_removed.csv --backend sqlite

  # Keep LAST occurrence (requires storing rows); uses sqlite row-store
  python dedup_smiles_inchi_large.py --input mp.csv --out mp_dedup.csv --keep last --backend sqlite
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import MolToInchiKey

KeepMode = Literal["first", "last"]
Backend = Literal["python", "sqlite"]
SmilesRep = Literal["canonical", "original"]


@dataclass(frozen=True)
class Config:
    smiles_col: str = "smiles"
    keep: KeepMode = "first"
    backend: Backend = "sqlite"
    smiles_rep: SmilesRep = "original"

    drop_invalid_smiles: bool = True
    keep_inchikey: bool = False

    chunksize: int = 100_000
    progress_every_chunks: int = 10


def _disable_rdkit_logs() -> None:
    try:
        rdBase.DisableLog("rdApp.*")
    except Exception:
        pass


def _compute_inchikey_and_canonical(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (inchikey, canonical_smiles) or (None, None) if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None, None
    if mol is None:
        return None, None
    try:
        ik = MolToInchiKey(mol)
    except Exception:
        return None, None
    if not ik:
        return None, None
    try:
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        can = None
    return ik, can


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")], errors="ignore")


def _open_removed_writer(path: Optional[Path]) -> tuple[Optional[csv.writer], Optional[object]]:
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(fh)
    w.writerow(["reason", "smiles", "inchi_key"])
    return w, fh


def _write_removed(w: Optional[csv.writer], reason: str, smiles: str, inchikey: Optional[str]) -> None:
    if w is None:
        return
    w.writerow([reason, smiles, "" if inchikey is None else inchikey])


# ----------------------------
# SQLite helpers (key-set mode for keep=first)
# ----------------------------
def _sqlite_init_seen(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")  # ~200MB cache (negative => KB)
    cur.execute("CREATE TABLE IF NOT EXISTS seen (inchi_key TEXT PRIMARY KEY);")
    conn.commit()


def _sqlite_try_add_seen(cur: sqlite3.Cursor, inchikey: str) -> bool:
    """
    Returns True if inserted (new), False if already existed.
    """
    cur.execute("INSERT OR IGNORE INTO seen(inchi_key) VALUES (?);", (inchikey,))
    return cur.rowcount == 1


# ----------------------------
# SQLite helpers (row-store mode for keep=last)
# ----------------------------
def _sqlite_init_rows(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS last_rows (
            inchi_key TEXT PRIMARY KEY,
            row_id    INTEGER NOT NULL,
            row_json  TEXT NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_last_rows_rowid ON last_rows(row_id);")
    conn.commit()


def _sqlite_upsert_last(cur: sqlite3.Cursor, inchikey: str, row_id: int, row_json: str) -> None:
    cur.execute(
        "INSERT OR REPLACE INTO last_rows(inchi_key, row_id, row_json) VALUES (?,?,?);",
        (inchikey, row_id, row_json),
    )


# ----------------------------
# Main logic
# ----------------------------
def dedup_large(cfg: Config, input_csv: Path, out_csv: Path, removed_csv: Optional[Path]) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input not found: {input_csv}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    removed_writer, removed_fh = _open_removed_writer(removed_csv)

    # Choose SQLite path if needed
    conn: Optional[sqlite3.Connection] = None
    cur: Optional[sqlite3.Cursor] = None
    db_file: Optional[str] = None
    delete_db_after = False

    if cfg.backend == "sqlite":
        tmp = tempfile.NamedTemporaryFile(prefix="dedup_seen_", suffix=".sqlite", delete=False)
        db_file = tmp.name
        tmp.close()
        delete_db_after = True
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        if cfg.keep == "first":
            _sqlite_init_seen(conn)
        else:
            _sqlite_init_rows(conn)

    seen_py: Optional[set[str]] = set() if (cfg.backend == "python" and cfg.keep == "first") else None
    if cfg.backend == "python" and cfg.keep == "last":
        raise ValueError("backend=python does not support keep=last (needs row-store). Use --backend sqlite.")

    # Stream input
    reader = pd.read_csv(
        input_csv,
        dtype=str,
        chunksize=cfg.chunksize,
        low_memory=False,
        keep_default_na=False,
    )

    wrote_header = False
    total_in = 0
    kept = 0
    invalid = 0
    dups = 0
    global_row_id = 0

    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as out_f:
            out_w = csv.writer(out_f)

            header_cols: list[str] = []

            for chunk_idx, df in enumerate(reader):
                df = _drop_unnamed(df)

                if cfg.smiles_col not in df.columns:
                    raise ValueError(
                        f"SMILES column '{cfg.smiles_col}' not found. Available: {list(df.columns)[:50]}"
                    )

                if not wrote_header:
                    header_cols = df.columns.tolist()
                    out_header = header_cols.copy()
                    if cfg.keep_inchikey:
                        out_header.append("inchi_key")
                    out_w.writerow(out_header)
                    wrote_header = True

                # Work with numpy array for faster row iteration
                cols = df.columns.tolist()
                sm_idx = cols.index(cfg.smiles_col)
                arr = df.to_numpy(dtype=object)

                # Begin transaction for sqlite
                if conn is not None:
                    conn.execute("BEGIN;")

                for i in range(arr.shape[0]):
                    total_in += 1
                    smi = str(arr[i, sm_idx]).strip()

                    if not smi:
                        invalid += 1
                        if cfg.drop_invalid_smiles:
                            _write_removed(removed_writer, "empty_smiles", smi, None)
                            continue

                    ik, can = _compute_inchikey_and_canonical(smi)
                    if ik is None:
                        invalid += 1
                        if cfg.drop_invalid_smiles:
                            _write_removed(removed_writer, "invalid_smiles_or_inchikey", smi, None)
                            continue
                        # If not dropping invalid, give it a unique-ish key so they don't all collapse
                        ik = f"INVALID_{global_row_id}"
                        can = smi

                    # Choose SMILES representation
                    if cfg.smiles_rep == "canonical" and can is not None and can != "":
                        arr[i, sm_idx] = can  # replace smiles with canonical

                    # Dedup logic
                    if cfg.keep == "first":
                        is_new = False
                        if cfg.backend == "python":
                            assert seen_py is not None
                            if ik not in seen_py:
                                seen_py.add(ik)
                                is_new = True
                        else:
                            assert cur is not None
                            is_new = _sqlite_try_add_seen(cur, ik)

                        if not is_new:
                            dups += 1
                            _write_removed(removed_writer, "duplicate_inchikey", smi, ik)
                            continue

                        # Write this row
                        row_out = arr[i].tolist()
                        if cfg.keep_inchikey:
                            row_out.append(ik)
                        out_w.writerow(row_out)
                        kept += 1

                    else:
                        # keep == "last" (row-store, write at end)
                        assert cur is not None
                        row = arr[i].tolist()
                        if cfg.keep_inchikey:
                            # store inchikey anyway; output will append too, but fine to keep consistent
                            pass
                        row_json = json.dumps(row, ensure_ascii=False)
                        _sqlite_upsert_last(cur, ik, global_row_id, row_json)
                        kept += 1  # provisional (will be exact at end, but useful progress)
                        global_row_id += 1
                        continue

                    global_row_id += 1

                # Commit sqlite chunk
                if conn is not None:
                    conn.commit()

                if (chunk_idx + 1) % cfg.progress_every_chunks == 0:
                    print(
                        f"[PROGRESS] chunks={chunk_idx+1} total_in={total_in:,} kept={kept:,} "
                        f"dups={dups:,} invalid={invalid:,}",
                        file=sys.stderr,
                    )

            # If keep=last, dump stored rows now (overwrite what we wrote earlier)
            if cfg.keep == "last":
                # Rewrite output from sqlite table (need to re-open file for rewriting)
                out_f.flush()

        if cfg.keep == "last":
            assert conn is not None
            # Rewrite out_csv from stored last_rows
            with open(out_csv, "w", newline="", encoding="utf-8") as out_f2:
                out_w2 = csv.writer(out_f2)

                # We need headers: reuse input headers by peeking first chunk again (cheap)
                peek = pd.read_csv(input_csv, dtype=str, nrows=1, keep_default_na=False)
                peek = _drop_unnamed(peek)
                header_cols = peek.columns.tolist()

                out_header = header_cols.copy()
                if cfg.keep_inchikey:
                    out_header.append("inchi_key")
                out_w2.writerow(out_header)

                cur2 = conn.cursor()
                cur2.execute("SELECT inchi_key, row_json FROM last_rows ORDER BY row_id ASC;")
                kept_exact = 0
                while True:
                    rows = cur2.fetchmany(10_000)
                    if not rows:
                        break
                    for ik, row_json in rows:
                        row = json.loads(row_json)
                        if cfg.keep_inchikey:
                            row.append(ik)
                        out_w2.writerow(row)
                        kept_exact += 1

            print(f"[OK] keep=last exact kept rows: {kept_exact:,}", file=sys.stderr)

    finally:
        if removed_fh is not None:
            removed_fh.close()

        if conn is not None:
            conn.close()

        if delete_db_after and db_file:
            try:
                os.remove(db_file)
            except Exception:
                pass

    print(f"[OK] input rows:  {total_in:,}")
    print(f"[OK] kept rows:   {kept:,}" if cfg.keep == "first" else f"[OK] processed rows (keep=last): {total_in:,}")
    print(f"[OK] duplicates:  {dups:,}")
    print(f"[OK] invalid:     {invalid:,}")
    print(f"[OK] output:      {out_csv}")
    if removed_csv is not None:
        print(f"[OK] removed:     {removed_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--removed", default="", help="Optional audit CSV for removed rows (invalid + duplicates)")
    p.add_argument("--smiles-col", default="smiles", help="SMILES column name (default: smiles)")
    p.add_argument("--keep", default="first", choices=["first", "last"], help="Keep first or last occurrence (default: first)")
    p.add_argument("--backend", default="sqlite", choices=["python", "sqlite"], help="Dedup backend (default: sqlite)")
    p.add_argument("--smiles-rep", default="original", choices=["original", "canonical"], help="Output SMILES representation")
    p.add_argument("--keep-inchikey", action="store_true", help="Append inchi_key column to output")
    p.add_argument("--no-drop-invalid-smiles", action="store_true", help="Do not drop invalid/empty SMILES (kept with synthetic keys)")
    p.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk (default: 100000)")
    p.add_argument("--progress-every", type=int, default=10, help="Progress print every N chunks (default: 10)")
    p.add_argument("--quiet-rdkit", action="store_true", help="Disable RDKit warnings/logs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.quiet_rdkit:
        _disable_rdkit_logs()

    cfg = Config(
        smiles_col=args.smiles_col,
        keep=args.keep,  # type: ignore[arg-type]
        backend=args.backend,  # type: ignore[arg-type]
        smiles_rep=args.smiles_rep,  # type: ignore[arg-type]
        drop_invalid_smiles=not bool(args.no_drop_invalid_smiles),
        keep_inchikey=bool(args.keep_inchikey),
        chunksize=int(args.chunksize),
        progress_every_chunks=int(args.progress_every),
    )

    input_csv = Path(args.input)
    out_csv = Path(args.out)
    removed_csv = Path(args.removed) if args.removed else None

    dedup_large(cfg, input_csv, out_csv, removed_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
