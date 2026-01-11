
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any

class PipelineMonitor:
    """
    Centralized Health Monitor for the Strategy Pipeline.
    Tracks critical system events, resource usage, and error rates.
    """
    _instance = None

    def __new__(cls, log_dir="logs"):
        if cls._instance is None:
            cls._instance = super(PipelineMonitor, cls).__new__(cls)
            cls._instance._initialize(log_dir)
        return cls._instance

    def _initialize(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "system_health.jsonl")
        
        # Setup standard Python logging as well for console output
        self.logger = logging.getLogger("PipelineMonitor")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def log_event(self, component: str, event_type: str, message: str, severity: str = "INFO", metadata: Dict[str, Any] = None):
        """
        Logs a health event.
        - severity: INFO, WARNING, ERROR, CRITICAL
        """
        timestamp = datetime.now().isoformat()
        
        entry = {
            "timestamp": timestamp,
            "component": component,
            "type": event_type,
            "severity": severity,
            "message": message,
            "metadata": metadata or {}
        }
        
        # 1. Write to JSONL
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
        # 2. Console (Filtered)
        if severity in ["WARNING", "ERROR", "CRITICAL"]:
            self.logger.warning(f"[{component}] {message}")
        else:
            self.logger.info(f"[{component}] {message}")

    def log_gpu_status(self, success: bool, details: str):
        severity = "INFO" if success else "WARNING"
        status = "GPU_ACTIVE" if success else "GPU_FALLBACK"
        self.log_event("VectorEngine", status, details, severity)

    def log_data_loading(self, symbol: str, source: str, success: bool):
        severity = "INFO" if success else "ERROR"
        status = "DATA_LOAD_SUCCESS" if success else "DATA_LOAD_FAIL"
        msg = f"Loaded {symbol} from {source}" if success else f"Failed to load {symbol} from {source}"
        self.log_event("DataHandler", status, msg, severity, {"symbol": symbol, "source": source})

if __name__ == "__main__":
    monitor = PipelineMonitor()
    monitor.log_event("Test", "INIT", "System initializing...")
    monitor.log_gpu_status(False, "Numba import failed. Switched to CPU.")
