import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from src.utils import logger
from src.mitre_mapper import MITREMapper


class ReportGenerator:
    """Generates professional, SOC-compliant threat intelligence reports with MITRE ATT&CK taxonomy."""

    def __init__(self, config, output_dir="outputs"):
        self.config = config
        self.output_dir = output_dir
        self.mitre = MITREMapper()

    def generate_report(self, results: Dict[str, Any], input_file: str, 
                        models_used: Dict[str, str], run_dir: str, 
                        df: Optional[pd.DataFrame] = None, 
                        drift_status: Optional[Dict[str, Any]] = None) -> str:
        
        report_dir = os.path.join(run_dir, "report")
        pred_dir = os.path.join(run_dir, "predictions")
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(os.path.join(pred_dir, "benign"), exist_ok=True)
        os.makedirs(os.path.join(pred_dir, "malicious"), exist_ok=True)
        os.makedirs(os.path.join(pred_dir, "suspicious"), exist_ok=True)
        
        filepath = os.path.join(report_dir, "Report.md")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base_name = os.path.basename(input_file)

        final_preds = results["final_preds"]
        reasons = results["reasons"]
        total_flows = len(final_preds)
        
        dos_count = final_preds.count("DoS/DDoS")
        critical_count = final_preds.count("Critical Anomaly")
        suspicious_count = final_preds.count("Suspicious")
        benign_count = final_preds.count("Benign")
        
        threats_total = dos_count + critical_count + suspicious_count
        
        dos_percent = (dos_count / total_flows) * 100 if total_flows > 0 else 0.0
        critical_percent = (critical_count / total_flows) * 100 if total_flows > 0 else 0.0
        suspicious_percent = (suspicious_count / total_flows) * 100 if total_flows > 0 else 0.0
        benign_percent = (benign_count / total_flows) * 100 if total_flows > 0 else 0.0
        threat_percent = (threats_total / total_flows) * 100 if total_flows > 0 else 0.0

        verdict = "SAFE"
        verdict_icon = "🟢 [SAFE]"
        verdict_desc = "No malicious traffic detected. All network profiles are normal."
        
        thresh_ratio = self.config.get("thresholds", {}).get("malicious_file_ratio", 0.30)
        
        if (dos_count / max(total_flows, 1)) > thresh_ratio:
            verdict = "MALICIOUS"
            verdict_icon = "🔴 [CRITICAL]"
            verdict_desc = f"Volumetric attack threshold exceeded ({dos_percent:.1f}% DoS/DDoS traffic). Action required."
        elif critical_count > 0:
            verdict = "CRITICAL"
            verdict_icon = "🟠 [CRITICAL]"
            verdict_desc = f"Unsupervised consensus flagged {critical_count} zero-day network anomalies. High priority."
        elif suspicious_count > 0:
            verdict = "SUSPICIOUS"
            verdict_icon = "🟡 [WARNING]"
            verdict_desc = f"Neuromorphic-tensor models flagged {suspicious_count} suspicious flows. Manual audit recommended."

        # --- Concept Drift Warnings ---
        drift_card = ""
        if drift_status and drift_status.get("drift_detected", False):
            sev = drift_status.get("severity", "WARNING")
            js = drift_status.get("js_divergence", 0.0)
            drift_card = f"""
> [!WARNING]
> **CONCEPT DRIFT DETECTED [{sev}]**
> Jensen-Shannon Divergence: `{js:.4f}` > Threshold: `{self.config.get('drift_detection', {}).get('js_threshold', 0.5)}`
> Network statistics have evolved significantly from the baseline training distribution. Retraining is recommended.
"""

        # --- Build Markdown ---
        content = f"""# Typhon IDS v4.0 -- Threat Intelligence Report

---

**Report Generated**: {timestamp}
**Input File Source**: `{base_name}`
**Analysis Framework**: Typhon Continuous-Time Neuromorphic-Tensor Fusion Architecture (CT-DGNN + ST3 + XGBoost)

---

## 1. Executive Summary

**Overall Verdict: {verdict_icon} {verdict}**

> {verdict_desc}
{drift_card}

| Traffic Metric | Flow Count | Percentage | Operational Status |
|---|---:|---:|---|
| **Total Flows Analyzed** | **{total_flows:,}** | **100.00%** | -- |
| Benign (Clean Traffic) | {benign_count:,} | {benign_percent:.2f}% | 🟢 NORMAL |
| Confirmed DoS/DDoS Attacks | {dos_count:,} | {dos_percent:.2f}% | 🔴 DANGER |
| Critical Zero-Day Anomalies | {critical_count:,} | {critical_percent:.2f}% | 🔴 DANGER |
| Suspicious Flow Anomalies | {suspicious_count:,} | {suspicious_percent:.2f}% | 🟡 AUDIT |
| **Total Security Alerts** | **{threats_total:,}** | **{threat_percent:.2f}%** | -- |

---

## 2. Dynamic Threat Severity Breakdown

| Severity Level | Flow Count | Tactical Attack Context |
|---|---:|---|
| **CRITICAL** (High Confidence) | {dos_count + critical_count:,} | Direct supervised threat signatures or triple unsupervised consensus. |
| **WARNING** (Medium Confidence) | {suspicious_count:,} | Single-dimensional neuromorphic or spatial tensor reconstruction anomalies. |
| **INFO** (Cleared Overrides) | {results.get('detection_stats', {}).get('override_benign', 0):,} | Validated broadcast, mDNS, ARP, or ICMP private keepalives. |

---

## 3. MITRE ATT&CK Enterprise Mapping

Threat signatures mapped strictly to the **MITRE ATT&CK Enterprise v14+** taxonomy:

| Mapping Category | Detections | Tactic | Technique | ID |
|---|---:|---|---|---|
"""
        # Collect unique mappings for this run
        mapped_tactics = {}
        for i in range(total_flows):
            if final_preds[i] != "Benign":
                reason = reasons[i]
                mitre_info = self.mitre.map_reason(reason)
                if mitre_info:
                    key = (mitre_info["tactic"], mitre_info["technique"], mitre_info["technique_id"])
                    mapped_tactics[key] = mapped_tactics.get(key, 0) + 1
                    
        if mapped_tactics:
            for (tactic, technique, tech_id), count in mapped_tactics.items():
                content += f"| {tactic} | {count:,} | {tactic} | {technique} | `{tech_id}` |\n"
        else:
            content += "| -- | -- | No adversarial mappings | -- | -- |\n"

        content += f"""
---

## 4. SOC Alerts Summary -- Confirmed Attacks (Top 30)

| Alert ID | Flow ID | Classification | Detection Reason | MITRE ATT&CK Context |
|---|---|---|---|---|
"""
        count_alerts = 0
        for i in range(total_flows):
            if final_preds[i] in ["DoS/DDoS", "Critical Anomaly"]:
                count_alerts += 1
                reason = reasons[i]
                mitre_info = self.mitre.map_reason(reason)
                tech_id = f"`{mitre_info['technique_id']}`" if mitre_info else "N/A"
                content += f"| {count_alerts} | Flow #{i} | {final_preds[i]} | {reason} | {tech_id} |\n"
                if count_alerts >= 30:
                    break
        if count_alerts == 0:
            content += "| -- | -- | No critical attacks identified | -- | -- |\n"

        content += f"""
---

## 5. SOC Alerts Summary -- Suspicious Anomalies (Top 30)

| Alert ID | Flow ID | Classification | Anomaly Metric | Severity |
|---|---|---|---|---|
"""
        count_sus = 0
        for i in range(total_flows):
            if final_preds[i] == "Suspicious":
                count_sus += 1
                reason = reasons[i]
                content += f"| {count_sus} | Flow #{i} | {reason} | WARNING | \n"
                if count_sus >= 30:
                    break
        if count_sus == 0:
            content += "| -- | -- | No suspicious anomalies identified | -- | -- |\n"

        # --- Ground Truth Accuracy ---
        if df is not None and "label" in df.columns:
            try:
                y_true = df["label"].astype(str).str.strip().str.upper().apply(
                    lambda x: 0 if x in ["BENIGN", "0", "0.0"] else 1
                ).values
                y_pred = np.array([0 if p == "Benign" else 1 for p in final_preds])
                
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                tn = int(((y_true == 0) & (y_pred == 0)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                fn = int(((y_true == 1) & (y_pred == 0)).sum())
                
                acc = (tp + tn) / total_flows if total_flows > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                
                content += f"""
---

## 6. Model Verification & Ground Truth Parity

> Label columns detected in input source. Auto-generating confusion matrix and accuracy validation metrics.

### 6.1 Confusion Matrix
| | Predicted Benign | Predicted Attack |
|---|---|---|
| **Actual Benign** | TN = {tn:,} | FP = {fp:,} |
| **Actual Attack** | FN = {fn:,} | TP = {tp:,} |

### 6.2 Performance Verification Metrics
| Metric | Value | Baseline Paper Metric | Parity State |
|---|---:|---:|---|
| **System Accuracy** | **{acc:.6f}** | 99.70% | {"✅ EXCEEDS" if acc >= 0.9970 else "❌ BELOW"} |
| **Recall (Detection Rate)** | **{rec:.6f}** | 99.95% | {"✅ EXCEEDS" if rec >= 0.9995 else "❌ BELOW"} |
| **Precision** | **{prec:.6f}** | 99.72% | {"✅ EXCEEDS" if prec >= 0.9972 else "❌ BELOW"} |
| **F1-Score** | **{f1:.6f}** | 99.83% | {"✅ EXCEEDS" if f1 >= 0.9983 else "❌ BELOW"} |
| **False Positive Rate** | **{fpr:.6f}** | 2.72% | {"✅ EXCEEDS" if fpr <= 0.0272 else "❌ BELOW"} |
| **False Negative Rate** | **{fnr:.6f}** | 0.05% | {"✅ EXCEEDS" if fnr <= 0.0005 else "❌ BELOW"} |
"""
            except Exception as e:
                logger.error(f"Parity stats generation failed: {e}")

        # Save MD Report
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # --- JSON SOC Summary ---
        json_path = filepath.replace(".md", ".json")
        json_data = {
            "timestamp": timestamp,
            "input_file": input_file,
            "verdict": verdict,
            "total_flows": total_flows,
            "metrics": {
                "dos_count": dos_count,
                "critical_anomaly_count": critical_count,
                "suspicious_anomaly_count": suspicious_count,
                "benign_count": benign_count,
                "total_threats": threats_total
            },
            "concept_drift": drift_status,
            "models_loaded": models_used
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"SIEM & Executive SOC Report generated: {filepath}")
        return filepath

    def save_predictions(self, df: pd.DataFrame, results: Dict[str, Any], run_dir: str):
        pred_dir = os.path.join(run_dir, "predictions")
        
        df_copy = df.copy()
        df_copy["Prediction"] = results["final_preds"]
        df_copy["Reason"] = results["reasons"]
        
        benign = df_copy[df_copy["Prediction"] == "Benign"]
        malicious = df_copy[df_copy["Prediction"] == "DoS/DDoS"]
        suspicious = df_copy[df_copy["Prediction"].isin(["Critical Anomaly", "Suspicious"])]
        
        if not benign.empty:
            benign.to_csv(os.path.join(pred_dir, "benign", "benign_flows.csv"), index=False)
            benign.to_json(os.path.join(pred_dir, "benign", "benign_flows.json"), orient="records", indent=2)
            
        if not malicious.empty:
            malicious.to_csv(os.path.join(pred_dir, "malicious", "malicious_flows.csv"), index=False)
            malicious.to_json(os.path.join(pred_dir, "malicious", "malicious_flows.json"), orient="records", indent=2)
            
        if not suspicious.empty:
            suspicious.to_csv(os.path.join(pred_dir, "suspicious", "suspicious_flows.csv"), index=False)
            suspicious.to_json(os.path.join(pred_dir, "suspicious", "suspicious_flows.json"), orient="records", indent=2)

        logger.info(f"Flow segregation complete: benign={len(benign):,}, malicious={len(malicious):,}, suspicious={len(suspicious):,}")
