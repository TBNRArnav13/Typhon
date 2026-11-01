import os
import psutil
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any, List, Optional
from src.utils import logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


try:
    CONFIG = load_config()
except Exception:
    CONFIG = {}


class SystemProfiler:
    """Helper to dynamically profile RAM and estimate row limits."""
    @staticmethod
    def estimate_rows(file_path: str) -> int:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        usable_bytes = mem.available * 0.5  # Reserve 50% for dynamic overhead
        
        bytes_per_row = 450  # ~33 double-precision features with pandas indexing overhead
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        file_rows_est = max(file_size // 120, 1000)
        
        max_rows_by_ram = int(usable_bytes / bytes_per_row)
        rows_to_read = min(max_rows_by_ram, file_rows_est)
        
        logger.info(f"System Profiler RAM-Aware Limits:")
        logger.info(f"  Available RAM: {available_gb:.2f} GB")
        logger.info(f"  File size: {file_size / (1024**2):.1f} MB")
        logger.info(f"  Estimated Rows: {file_rows_est:,}")
        logger.info(f"  RAM Limit Cap: {max_rows_by_ram:,}")
        logger.info(f"  -> Selected row count: {rows_to_read:,}")
        
        return max(rows_to_read, 1000)


def load_csv(filepath: str, max_rows: Optional[int] = None, skip_rows: int = 0) -> Optional[pd.DataFrame]:
    """Loads a CSV file cleanly with memory-efficient chunks if the file is massive."""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None

    try:
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # Estimate rows if set to auto or unspecified
        if max_rows is None:
            cfg_rows = CONFIG.get("data_loading", {}).get("max_rows_per_file", "auto")
            if cfg_rows == "auto":
                max_rows = SystemProfiler.estimate_rows(filepath)
            elif cfg_rows is not None:
                max_rows = int(cfg_rows)

        logger.info(f"Loading CSV {filepath} ({file_size_mb:.1f} MB, max_rows={max_rows or 'all'})...")
        
        # Fast chunked reading for massive files (>500MB)
        if file_size_mb > 500 and max_rows is not None:
            chunk_size = 50_000
            chunks = []
            rows_read = 0
            
            reader = pd.read_csv(filepath, chunksize=chunk_size, on_bad_lines="skip", low_memory=True)
            for chunk in reader:
                remaining = max_rows - rows_read
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining]
                chunks.append(chunk)
                rows_read += len(chunk)
            if hasattr(reader, 'close'):
                reader.close()
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        else:
            if skip_rows > 0:
                df = pd.read_csv(filepath, nrows=max_rows, skiprows=range(1, skip_rows + 1), 
                                 on_bad_lines="skip", low_memory=False)
            else:
                df = pd.read_csv(filepath, nrows=max_rows, on_bad_lines="skip", low_memory=False)

        # Standardize column headers
        df.columns = df.columns.str.strip().str.lower()
        
        # Keep only the 33 target features + label
        selected_features = [f.strip().lower() for f in CONFIG.get("selected_features", [])]
        if "label" in df.columns and "label" not in selected_features:
            selected_features.append("label")
            
        existing_cols = [c for c in selected_features if c in df.columns]
        df = df[existing_cols]
        
        # Null / Inf sanitization
        df.replace([float("inf"), float("-inf")], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        logger.info(f"Loaded DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed to load CSV {filepath}: {e}")
        return None


def combine_datasets(benign_path: str, dos_dir: str) -> Optional[pd.DataFrame]:
    """Combines benign and malicious CSV files into a unified corpus."""
    dfs = []
    
    benign_df = load_csv(benign_path)
    if benign_df is not None:
        dfs.append(benign_df)
        
    if os.path.exists(dos_dir):
        for f in sorted(os.listdir(dos_dir)):
            if f.endswith(".csv"):
                fpath = os.path.join(dos_dir, f)
                df = load_csv(fpath)
                if df is not None:
                    dfs.append(df)
                    
    if not dfs:
        return None
        
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    logger.info(f"Aggregated unified dataset shape: {combined.shape}")
    return combined
