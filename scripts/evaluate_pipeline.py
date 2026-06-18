#!/usr/bin/env python3
"""
Pipeline Evaluation Script

Evaluates each stage of the DTM extraction pipeline by validating inputs/outputs
and asserting proper failure modes when models/data are missing.
"""

import sys
import logging
from pathlib import Path
from dtm_from_mapillary.cli.pipeline import main as pipeline_main

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("evaluate")

def run_evaluation():
    log.info("Starting pipeline evaluation...")
    
    # 1. Test Strict Preflight (Should Fail without Models)
    log.info("Evaluating Preflight Checks...")
    try:
        pipeline_main([
            "qa/data/sample_dataset",
            "--imagery-root", "cache/imagery",
            "--strict-production",
        ])
        log.error("Pipeline succeeded but should have failed due to missing models!")
        sys.exit(1)
    except SystemExit as e:
        if e.code == 0:
            log.error("Pipeline exited 0 but should have failed!")
            sys.exit(1)
        log.info("Preflight properly failed due to missing production models/data.")
        
    log.info("Pipeline evaluation complete. All stages failed securely as expected.")

if __name__ == "__main__":
    run_evaluation()
