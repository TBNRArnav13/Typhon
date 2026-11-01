import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.utils import logger


class ConceptDriftDetector:
    """
    Jensen-Shannon Divergence Concept Drift Detection Engine.
    
    Computes statistical shifts between training baseline feature distributions and
    real-time testing/production distributions over normalized probability mass functions.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        drift_cfg = config.get("drift_detection", {})
        self.enabled = drift_cfg.get("enabled", True)
        self.js_threshold = drift_cfg.get("js_threshold", 0.5)
        self.n_bins = drift_cfg.get("n_bins", 50)
        self.selected_features = [f.strip().lower() for f in config.get("selected_features", [])]
        
        self.baseline_pmfs: Dict[str, np.ndarray] = {}
        self.baseline_bins: Dict[str, np.ndarray] = {}

    def fit_baseline(self, df_baseline: pd.DataFrame):
        """Builds normalized PMF references from training dataset."""
        if not self.enabled:
            return
            
        logger.info("Calibrating Concept Drift Detector reference PMFs...")
        for col in self.selected_features:
            if col not in df_baseline.columns:
                continue
                
            vals = df_baseline[col].values
            # Compute histogram bin edges and densities over baseline
            counts, bin_edges = np.histogram(vals, bins=self.n_bins, density=False)
            
            # Convert counts to a normalized Probability Mass Function (PMF)
            pmf = counts / max(counts.sum(), 1)
            # Add a small epsilon to avoid divide-by-zero or zero log metrics in KL computation
            pmf = pmf + 1e-9
            pmf = pmf / pmf.sum()
            
            self.baseline_pmfs[col] = pmf
            self.baseline_bins[col] = bin_edges
            
        logger.info(f"Concept Drift calibration complete. Tracked features: {len(self.baseline_pmfs)}")

    def _kullback_leibler_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Computes symmetric Kullback-Leibler divergence between two PMFs."""
        # Assert equal length
        p = p + 1e-9
        p = p / p.sum()
        q = q + 1e-9
        q = q / q.sum()
        return float(np.sum(p * np.log2(p / q)))

    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Computes bounded Jensen-Shannon divergence (0.0 = identical, 1.0 = disjoint)."""
        m = 0.5 * (p + q)
        kl_pm = self._kullback_leibler_divergence(p, m)
        kl_qm = self._kullback_leibler_divergence(q, m)
        return float(0.5 * kl_pm + 0.5 * kl_qm)

    def detect_drift(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes Jensen-Shannon divergence across features and asserts drift status.
        """
        if not self.enabled or not self.baseline_pmfs:
            return {"drift_detected": False, "js_divergence": 0.0, "severity": "NONE"}
            
        js_scores = []
        
        for col in self.selected_features:
            if col not in df_test.columns or col not in self.baseline_pmfs:
                continue
                
            vals = df_test[col].values
            bin_edges = self.baseline_bins[col]
            pmf_base = self.baseline_pmfs[col]
            
            # Compute test histogram using matching baseline bin boundaries
            counts, _ = np.histogram(vals, bins=bin_edges)
            pmf_test = counts / max(counts.sum(), 1)
            pmf_test = pmf_test + 1e-9
            pmf_test = pmf_test / pmf_test.sum()
            
            # Calculate Jensen-Shannon divergence
            js = self._jensen_shannon_divergence(pmf_base, pmf_test)
            js_scores.append(js)
            
        if not js_scores:
            return {"drift_detected": False, "js_divergence": 0.0, "severity": "NONE"}
            
        avg_js = float(np.mean(js_scores))
        drift_detected = avg_js > self.js_threshold
        
        severity = "NONE"
        if drift_detected:
            severity = "CRITICAL" if avg_js > (self.js_threshold * 1.5) else "WARNING"
            
        logger.info(f"Concept Drift Assessment: JS-Divergence={avg_js:.4f} (Threshold={self.js_threshold})")
        if drift_detected:
            logger.warning(f"!!! CONCEPT DRIFT DETECTED [{severity}] !!! Network statistics shifted significantly.")
            
        # Only zip with features that were actually computed (avoids misalignment when features are skipped)
        computed_features = [f for f in self.selected_features
                           if f in df_test.columns and f in self.baseline_pmfs]
        return {
            "drift_detected": drift_detected,
            "js_divergence": avg_js,
            "severity": severity,
            "js_scores_by_feature": {col: float(score) for col, score in zip(computed_features, js_scores)}
        }
