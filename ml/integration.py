"""Integration helpers for learned uncertainty calibration in the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..common_core import GroundPoint
from ..ml.uncertainty_calibration import (
    UncertaintyCalibrator,
    UncertaintyFeatures,
    prepare_training_data_from_consensus,
)

log = logging.getLogger(__name__)


def train_uncertainty_model_from_consensus(
    consensus_points: Sequence[Dict[str, object]],
    source_points: Dict[str, Sequence[Dict[str, object]]],
    model_path: Optional[Path | str] = None,
    backend: str = "sklearn",
) -> UncertaintyCalibrator:
    """Train uncertainty calibration model from consensus validation.

    Parameters
    ----------
    consensus_points : list of dict
        Consensus points with aggregated metadata
    source_points : dict
        Mapping of source name → list of points from that source
    model_path : Path or str, optional
        Path to save trained model
    backend : str
        ML backend: "sklearn", "xgboost", or "simple"

    Returns
    -------
    UncertaintyCalibrator
        Trained calibration model
    """
    log.info("Preparing training data from %d consensus points", len(consensus_points))

    features, true_errors = prepare_training_data_from_consensus(
        consensus_points, source_points
    )

    if len(features) < 50:
        log.warning(
            "Only %d training samples available; consider using more data or fallback to heuristic",
            len(features),
        )

    calibrator = UncertaintyCalibrator(backend=backend)
    metrics = calibrator.train(features, true_errors, val_split=0.2)

    log.info("Uncertainty model training complete:")
    log.info("  MAE: %.4f m", metrics["mae"])
    log.info("  RMSE: %.4f m", metrics["rmse"])
    log.info("  R²: %.4f", metrics["r2"])
    log.info("  Calibration Error: %.4f m", metrics["calibration_error"])

    if model_path:
        calibrator.save(model_path)
        log.info("Saved calibration model to %s", model_path)

    return calibrator


def apply_learned_uncertainty(
    ground_points: Sequence[GroundPoint],
    calibrator: UncertaintyCalibrator,
) -> List[GroundPoint]:
    """Replace heuristic uncertainties with learned predictions.

    Parameters
    ----------
    ground_points : list of GroundPoint
        Points with heuristic uncertainty estimates
    calibrator : UncertaintyCalibrator
        Trained calibration model

    Returns
    -------
    list of GroundPoint
        Points with calibrated uncertainties
    """
    if not ground_points:
        return []

    # Extract features from ground points
    features: List[UncertaintyFeatures] = []
    for gp in ground_points:
        feat = UncertaintyFeatures(
            tri_angle_deg=gp.tri_angle_deg if gp.tri_angle_deg is not None else 0.0,
            view_count=gp.view_count,
            sem_prob=gp.sem_prob,
            base_uncertainty=gp.uncertainty_m,
            min_distance=10.0,  # Would need camera centers for exact value
            max_baseline=5.0,  # Would need camera centers for exact value
            mask_variance=0.05,  # Would need mask values for exact value
            local_density=2.0,  # Would need neighborhood for exact value
            method=gp.method,
        )
        features.append(feat)

    # Predict calibrated uncertainties
    calibrated_uncertainties = calibrator.predict(features)

    # Create new GroundPoint objects with calibrated uncertainties
    calibrated_points: List[GroundPoint] = []
    for gp, new_unc in zip(ground_points, calibrated_uncertainties):
        calibrated_points.append(
            GroundPoint(
                x=gp.x,
                y=gp.y,
                z=gp.z,
                method=gp.method,
                seq_id=gp.seq_id,
                image_ids=gp.image_ids,
                view_count=gp.view_count,
                sem_prob=gp.sem_prob,
                tri_angle_deg=gp.tri_angle_deg,
                uncertainty_m=float(new_unc),
            )
        )

    log.info("Applied learned uncertainty to %d points", len(calibrated_points))
    return calibrated_points


def load_or_train_calibrator(
    model_path: Path | str,
    consensus_points: Optional[Sequence[Dict[str, object]]] = None,
    source_points: Optional[Dict[str, Sequence[Dict[str, object]]]] = None,
    backend: str = "sklearn",
) -> UncertaintyCalibrator:
    """Load existing calibrator or train new one if not found.

    Parameters
    ----------
    model_path : Path or str
        Path to saved model
    consensus_points : list of dict, optional
        Training data (consensus points)
    source_points : dict, optional
        Training data (source points)
    backend : str
        ML backend if training needed

    Returns
    -------
    UncertaintyCalibrator
        Loaded or newly trained calibrator
    """
    model_path = Path(model_path)

    if model_path.exists():
        log.info("Loading existing uncertainty calibrator from %s", model_path)
        calibrator = UncertaintyCalibrator()
        calibrator.load(model_path)
        return calibrator

    log.info("No existing model found; training new calibrator")

    if consensus_points is None or source_points is None:
        raise ValueError(
            "consensus_points and source_points required for training but not provided"
        )

    return train_uncertainty_model_from_consensus(
        consensus_points, source_points, model_path, backend
    )
