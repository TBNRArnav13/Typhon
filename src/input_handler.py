import os
from typing import Optional
import pandas as pd
from src.utils import logger
from src.data_loader import load_csv


class InputHandler:
    """Validates files and dynamically routes CSV or raw PCAP files to the correct parsing sub-pipeline."""

    def __init__(self, config):
        self.config = config
        self.supported_extensions = [".csv", ".pcap", ".pcapng"]

    def validate_file(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return False
            
        _, ext = os.path.splitext(filepath)
        if ext.lower() not in self.supported_extensions:
            logger.error(f"Unsupported file format: {ext}")
            return False
            
        return True

    def process_file(self, filepath: str) -> Optional[pd.DataFrame]:
        if not self.validate_file(filepath):
            return None

        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext == ".csv":
            logger.info(f"Routing to memory-efficient CSV loader: {filepath}")
            return load_csv(filepath)
            
        elif ext in [".pcap", ".pcapng"]:
            logger.info(f"Routing to high-fidelity packet extraction engine: {filepath}")
            from src.feature_extractor import FeatureExtractor
            extractor = FeatureExtractor(self.config)
            df = extractor.extract_features(filepath)
            
            if df is not None and not df.empty:
                logger.info(f"Flow assembly completed. Total bidirectional flows extracted: {len(df)}")
                return df
            else:
                logger.error("Feature extraction returned an empty DataFrame.")
                return None
                
        return None
