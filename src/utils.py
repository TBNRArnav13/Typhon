import os
import sys
import logging
import torch

# Setup centralized logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Typhon")


def get_device() -> torch.device:
    """
    Robust device loader with a 3-retry fallback logic for CUDA.
    Flushes cache on every failed attempt to mitigate transient memory issues.
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if torch.cuda.is_available():
                # Test allocations
                device = torch.device("cuda:0")
                _ = torch.zeros(1, device=device)
                logger.info(f"CUDA successfully initialized on attempt {attempt}/3.")
                return device
        except Exception as e:
            logger.warning(f"CUDA initialization failed on attempt {attempt}/3: {e}")
            torch.cuda.empty_cache()
            
    logger.warning("CUDA unavailable after 3 attempts. Falling back to CPU execution.")
    return torch.device("cpu")


def get_xgboost_device() -> str:
    """Returns 'cuda' if GPU available, otherwise 'cpu' for XGBoost."""
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
