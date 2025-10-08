"""
Learned uncertainty calibration for ground point reliability estimation.

This module implements a machine learning approach to predict point-level
uncertainty based on geometric and semantic features. The model is trained
on consensus-validated points where we can compute actual errors from
multi-source agreement.

Features used:
- Triangulation angle (degrees)
- View count
- Semantic probability (ground mask confidence)
- Base method uncertainty (from mono-depth, SfM, etc.)
- Distance to nearest camera
- Inter-camera baseline
- Ground mask variance across views
- Local point density

The calibrated uncertainty better reflects actual error distributions and
enables more reliable fusion and confidence mapping.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


class SimpleLinearModel:
    """Simple picklable linear regression model."""

    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights


@dataclass
class UncertaintyFeatures:
    """Feature vector for a single ground point."""

    tri_angle_deg: float
    view_count: int
    sem_prob: float
    base_uncertainty: float
    min_distance: float
    max_baseline: float
    mask_variance: float
    local_density: float
    method: str  # "opensfm"|"colmap"|"mono"|"plane_sweep"

    def to_array(self) -> np.ndarray:
        """Convert to numerical feature vector (excluding method)."""
        return np.array(
            [
                self.tri_angle_deg,
                float(self.view_count),
                self.sem_prob,
                self.base_uncertainty,
                self.min_distance,
                self.max_baseline,
                self.mask_variance,
                self.local_density,
            ],
            dtype=np.float32,
        )

    @property
    def method_code(self) -> int:
        """One-hot encoding helper for method."""
        mapping = {"opensfm": 0, "colmap": 1, "mono": 2, "plane_sweep": 3, "anchor": 4}
        return mapping.get(self.method.lower(), 0)


class UncertaintyCalibrator:
    """Learned model for uncertainty estimation.

    Uses a simple ensemble of gradient boosting models (or random forest)
    to predict uncertainty from point features. The model is trained on
    consensus points where we can measure actual 3D errors.
    """

    def __init__(self, backend: str = "sklearn"):
        """Initialize calibrator.

        Parameters
        ----------
        backend : str
            ML backend to use: "sklearn" (default), "xgboost", or "simple"
        """
        self.backend = backend
        self.model = None
        self.scaler = None
        self.feature_names = [
            "tri_angle_deg",
            "view_count",
            "sem_prob",
            "base_uncertainty",
            "min_distance",
            "max_baseline",
            "mask_variance",
            "local_density",
        ]
        self._is_trained = False

    def train(
        self,
        features: Sequence[UncertaintyFeatures],
        true_errors: Sequence[float],
        val_split: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """Train the uncertainty model.

        Parameters
        ----------
        features : list of UncertaintyFeatures
            Feature vectors for training samples
        true_errors : list of float
            Actual 3D errors (in meters) for each sample. Typically computed
            from consensus agreement or external reference data.
        val_split : float
            Fraction of data to use for validation
        random_state : int
            RNG seed for reproducibility

        Returns
        -------
        dict
            Training metrics (MAE, R2, calibration error)
        """
        if len(features) != len(true_errors):
            raise ValueError("Features and true_errors must have same length")

        if len(features) < 100:
            log.warning("Training with < 100 samples; model may underfit")

        X = np.vstack([f.to_array() for f in features])
        y = np.asarray(true_errors, dtype=np.float32)

        # Add method one-hot encoding
        method_codes = np.array([f.method_code for f in features], dtype=np.int32)
        method_onehot = np.eye(5, dtype=np.float32)[method_codes]
        X = np.hstack([X, method_onehot])

        # Train/val split
        rng = np.random.default_rng(random_state)
        n = X.shape[0]
        indices = rng.permutation(n)
        split_idx = int(n * (1.0 - val_split))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Normalize features
        self.scaler = self._fit_scaler(X_train)
        X_train = self._transform(X_train)
        X_val = self._transform(X_val)

        # Train model
        if self.backend == "sklearn":
            self.model = self._train_sklearn(X_train, y_train)
        elif self.backend == "xgboost":
            self.model = self._train_xgboost(X_train, y_train)
        else:  # "simple"
            self.model = self._train_simple(X_train, y_train)

        self._is_trained = True

        # Validation metrics
        y_pred = self.model.predict(X_val)
        metrics = self._compute_metrics(y_val, y_pred)
        metrics["train_samples"] = len(train_idx)
        metrics["val_samples"] = len(val_idx)

        log.info(
            "Uncertainty calibration trained: MAE=%.4f, R²=%.4f",
            metrics["mae"],
            metrics["r2"],
        )
        return metrics

    def predict(self, features: Sequence[UncertaintyFeatures]) -> np.ndarray:
        """Predict calibrated uncertainties.

        Parameters
        ----------
        features : list of UncertaintyFeatures
            Feature vectors for points to predict

        Returns
        -------
        np.ndarray
            Predicted uncertainties (meters), shape (N,)
        """
        if not self._is_trained:
            log.warning("Model not trained; returning base uncertainties")
            return np.array([f.base_uncertainty for f in features], dtype=np.float32)

        X = np.vstack([f.to_array() for f in features])
        method_codes = np.array([f.method_code for f in features], dtype=np.int32)
        method_onehot = np.eye(5, dtype=np.float32)[method_codes]
        X = np.hstack([X, method_onehot])
        X = self._transform(X)

        predictions = self.model.predict(X).astype(np.float32)
        # Clip predictions to reasonable uncertainty range [0.05, 0.6] meters
        return np.clip(predictions, 0.05, 0.6)

    def save(self, path: Path | str) -> None:
        """Persist model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "backend": self.backend,
                    "feature_names": self.feature_names,
                    "is_trained": self._is_trained,
                },
                f,
            )
        log.info("Saved uncertainty calibrator to %s", path)

    def load(self, path: Path | str) -> None:
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with path.open("rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.backend = data["backend"]
        self.feature_names = data["feature_names"]
        self._is_trained = data["is_trained"]
        log.info("Loaded uncertainty calibrator from %s", path)

    def _fit_scaler(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Fit standard scaler (mean/std normalization)."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std < 1e-6] = 1.0  # Avoid division by zero
        return {"mean": mean, "std": std}

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Apply feature scaling."""
        if self.scaler is None:
            return X
        return (X - self.scaler["mean"]) / self.scaler["std"]

    def _train_sklearn(self, X: np.ndarray, y: np.ndarray):
        """Train sklearn RandomForestRegressor."""
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            log.warning("scikit-learn not available; falling back to simple model")
            return self._train_simple(X, y)

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost regressor."""
        try:
            import xgboost as xgb
        except ImportError:
            log.warning("xgboost not available; falling back to sklearn")
            return self._train_sklearn(X, y)

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X, y)
        return model

    def _train_simple(self, X: np.ndarray, y: np.ndarray):
        """Simple linear regression fallback."""
        # Ridge regression: w = (X^T X + λI)^-1 X^T y
        lambda_reg = 0.1
        XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
        Xty = X.T @ y
        w = np.linalg.solve(XtX, Xty)

        return SimpleLinearModel(w)

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))

        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-9))

        # Calibration error (ideal: predicted ≈ actual error)
        abs_diff = np.abs(y_pred - y_true)
        calibration_error = float(np.mean(abs_diff))

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "calibration_error": calibration_error,
        }


def extract_features_from_ground_point(
    point: Dict[str, object],
    camera_centers: Sequence[np.ndarray],
    mask_values: Sequence[float],
    local_points: Sequence[np.ndarray],
) -> UncertaintyFeatures:
    """Extract features from a ground point for calibration.

    Parameters
    ----------
    point : dict
        Ground point with keys: x, y, z, method, uncertainty, sem_prob, etc.
    camera_centers : list of np.ndarray
        Supporting camera center positions (3D)
    mask_values : list of float
        Ground mask probabilities from supporting views
    local_points : list of np.ndarray
        Nearby points within 2m for density estimation

    Returns
    -------
    UncertaintyFeatures
        Feature vector for the point
    """
    pt_xyz = np.array([point["x"], point["y"], point["z"]], dtype=np.float64)

    # Triangulation angle
    tri_angle = float(point.get("tri_angle_deg", 0.0))

    # View count
    view_count = int(point.get("support", 1))

    # Semantic probability
    sem_prob = float(point.get("sem_prob", 0.7))

    # Base uncertainty
    base_uncertainty = float(point.get("uncertainty", 0.25))

    # Distance to nearest camera
    if camera_centers:
        distances = [np.linalg.norm(pt_xyz - c) for c in camera_centers]
        min_distance = float(np.min(distances))
    else:
        min_distance = 10.0

    # Maximum baseline between cameras
    if len(camera_centers) >= 2:
        baselines = []
        for i in range(len(camera_centers)):
            for j in range(i + 1, len(camera_centers)):
                baselines.append(np.linalg.norm(camera_centers[i] - camera_centers[j]))
        max_baseline = float(np.max(baselines)) if baselines else 1.0
    else:
        max_baseline = 1.0

    # Mask variance across views
    if mask_values:
        mask_variance = float(np.var(mask_values))
    else:
        mask_variance = 0.0

    # Local point density
    if local_points:
        # Count points within 2m radius
        local_points_arr = np.vstack(local_points)
        dists = np.linalg.norm(local_points_arr - pt_xyz, axis=1)
        within_2m = np.sum(dists <= 2.0)
        local_density = float(within_2m / (np.pi * 4.0))  # points per m²
    else:
        local_density = 0.5

    method = str(point.get("method", "unknown"))

    return UncertaintyFeatures(
        tri_angle_deg=tri_angle,
        view_count=view_count,
        sem_prob=sem_prob,
        base_uncertainty=base_uncertainty,
        min_distance=min_distance,
        max_baseline=max_baseline,
        mask_variance=mask_variance,
        local_density=local_density,
        method=method,
    )


def prepare_training_data_from_consensus(
    consensus_points: Sequence[Dict[str, object]],
    source_points: Dict[str, Sequence[Dict[str, object]]],
) -> Tuple[List[UncertaintyFeatures], List[float]]:
    """Prepare training data from consensus validation.

    For each consensus point, we compute the actual error as the standard
    deviation across contributing sources. This gives us ground truth for
    the uncertainty model.

    Parameters
    ----------
    consensus_points : list of dict
        Consensus points with aggregated metadata
    source_points : dict
        Mapping of source name → list of points from that source

    Returns
    -------
    features : list of UncertaintyFeatures
        Feature vectors for training
    errors : list of float
        True errors (meters) for each point
    """
    features: List[UncertaintyFeatures] = []
    errors: List[float] = []

    for cpt in consensus_points:
        cx, cy = float(cpt["x"]), float(cpt["y"])

        # Find source points near this consensus point
        source_zs: List[float] = []
        for source_name, src_pts in source_points.items():
            for spt in src_pts:
                dx = float(spt["x"]) - cx
                dy = float(spt["y"]) - cy
                if dx**2 + dy**2 < 0.25**2:  # within 0.25m
                    source_zs.append(float(spt["z"]))

        if len(source_zs) < 2:
            continue  # Need multiple sources for error estimation

        # True error = std dev of heights
        true_error = float(np.std(source_zs))

        # Extract features (simplified - full implementation would need camera centers etc.)
        feat = UncertaintyFeatures(
            tri_angle_deg=float(cpt.get("tri_angle_deg", 0.0)),
            view_count=int(cpt.get("support", 1)),
            sem_prob=float(cpt.get("sem_prob", 0.7)),
            base_uncertainty=float(cpt.get("uncertainty", 0.25)),
            min_distance=10.0,  # Placeholder
            max_baseline=5.0,  # Placeholder
            mask_variance=0.05,  # Placeholder
            local_density=2.0,  # Placeholder
            method=str(cpt.get("method", "consensus")),
        )

        features.append(feat)
        errors.append(true_error)

    log.info("Prepared %d training samples from consensus", len(features))
    return features, errors
