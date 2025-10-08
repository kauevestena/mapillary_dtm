"""Tests for learned uncertainty calibration."""

from __future__ import annotations

import numpy as np
import pytest

from ml.uncertainty_calibration import (
    UncertaintyCalibrator,
    UncertaintyFeatures,
    extract_features_from_ground_point,
    prepare_training_data_from_consensus,
)


def test_uncertainty_features_to_array():
    """Test feature vector conversion."""
    feat = UncertaintyFeatures(
        tri_angle_deg=15.0,
        view_count=3,
        sem_prob=0.85,
        base_uncertainty=0.25,
        min_distance=8.5,
        max_baseline=4.2,
        mask_variance=0.03,
        local_density=2.5,
        method="opensfm",
    )

    arr = feat.to_array()
    assert arr.shape == (8,)
    assert arr[0] == 15.0  # tri_angle
    assert arr[1] == 3.0  # view_count
    assert arr[2] == 0.85  # sem_prob


def test_uncertainty_features_method_code():
    """Test method encoding."""
    feat_opensfm = UncertaintyFeatures(
        tri_angle_deg=10.0,
        view_count=2,
        sem_prob=0.8,
        base_uncertainty=0.2,
        min_distance=5.0,
        max_baseline=3.0,
        mask_variance=0.02,
        local_density=1.5,
        method="opensfm",
    )
    feat_colmap = UncertaintyFeatures(
        tri_angle_deg=10.0,
        view_count=2,
        sem_prob=0.8,
        base_uncertainty=0.2,
        min_distance=5.0,
        max_baseline=3.0,
        mask_variance=0.02,
        local_density=1.5,
        method="colmap",
    )

    assert feat_opensfm.method_code == 0
    assert feat_colmap.method_code == 1


def test_calibrator_simple_backend():
    """Test simple linear regression backend."""
    calibrator = UncertaintyCalibrator(backend="simple")

    # Generate synthetic training data
    rng = np.random.default_rng(42)
    n_samples = 200

    features = []
    true_errors = []

    for _ in range(n_samples):
        tri_angle = float(rng.uniform(1.0, 30.0))
        view_count = int(rng.integers(2, 6))
        sem_prob = float(rng.uniform(0.6, 0.95))
        base_unc = float(rng.uniform(0.1, 0.4))

        # True error depends on features (synthetic relationship)
        true_error = (
            base_unc
            * (30.0 / max(tri_angle, 1.0))
            * (3.0 / view_count)
            * (1.5 - sem_prob)
        )
        true_error = max(0.05, min(0.6, true_error + rng.normal(0, 0.05)))

        feat = UncertaintyFeatures(
            tri_angle_deg=tri_angle,
            view_count=view_count,
            sem_prob=sem_prob,
            base_uncertainty=base_unc,
            min_distance=float(rng.uniform(5.0, 15.0)),
            max_baseline=float(rng.uniform(2.0, 8.0)),
            mask_variance=float(rng.uniform(0.01, 0.1)),
            local_density=float(rng.uniform(1.0, 5.0)),
            method="opensfm",
        )
        features.append(feat)
        true_errors.append(true_error)

    # Train
    metrics = calibrator.train(features, true_errors, val_split=0.2)

    # Check metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["train_samples"] > 0
    assert metrics["val_samples"] > 0

    # Predict
    predictions = calibrator.predict(features[:10])
    assert predictions.shape == (10,)
    assert np.all(predictions > 0.0)
    assert np.all(predictions < 1.0)


def test_calibrator_sklearn_backend():
    """Test sklearn RandomForest backend (if available)."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        pytest.skip("scikit-learn not available")

    calibrator = UncertaintyCalibrator(backend="sklearn")

    # Small synthetic dataset
    rng = np.random.default_rng(123)
    n_samples = 150

    features = []
    true_errors = []

    for _ in range(n_samples):
        tri_angle = float(rng.uniform(2.0, 25.0))
        view_count = int(rng.integers(2, 5))
        sem_prob = float(rng.uniform(0.65, 0.9))
        base_unc = float(rng.uniform(0.15, 0.35))

        true_error = 0.1 + 0.3 * (1.0 / max(tri_angle, 1.0)) + 0.1 * (1.0 / view_count)
        true_error = np.clip(true_error + rng.normal(0, 0.03), 0.05, 0.5)

        feat = UncertaintyFeatures(
            tri_angle_deg=tri_angle,
            view_count=view_count,
            sem_prob=sem_prob,
            base_uncertainty=base_unc,
            min_distance=float(rng.uniform(6.0, 12.0)),
            max_baseline=float(rng.uniform(3.0, 6.0)),
            mask_variance=float(rng.uniform(0.02, 0.08)),
            local_density=float(rng.uniform(1.5, 4.0)),
            method="colmap",
        )
        features.append(feat)
        true_errors.append(true_error)

    metrics = calibrator.train(features, true_errors, val_split=0.25)

    assert metrics["mae"] < 0.5  # Reasonable error
    assert calibrator._is_trained

    # Predict on new data
    test_feat = UncertaintyFeatures(
        tri_angle_deg=10.0,
        view_count=3,
        sem_prob=0.8,
        base_uncertainty=0.25,
        min_distance=8.0,
        max_baseline=4.0,
        mask_variance=0.04,
        local_density=2.5,
        method="colmap",
    )
    pred = calibrator.predict([test_feat])
    assert pred.shape == (1,)
    assert 0.05 < pred[0] < 0.6


def test_calibrator_save_load(tmp_path):
    """Test model persistence."""
    calibrator = UncertaintyCalibrator(backend="simple")

    # Train on minimal data
    rng = np.random.default_rng(99)
    features = [
        UncertaintyFeatures(
            tri_angle_deg=float(rng.uniform(5, 20)),
            view_count=int(rng.integers(2, 4)),
            sem_prob=0.8,
            base_uncertainty=0.2,
            min_distance=10.0,
            max_baseline=5.0,
            mask_variance=0.05,
            local_density=2.0,
            method="opensfm",
        )
        for _ in range(100)
    ]
    true_errors = [0.2 + rng.normal(0, 0.05) for _ in range(100)]

    calibrator.train(features, true_errors, val_split=0.2)

    # Save
    model_path = tmp_path / "calibrator.pkl"
    calibrator.save(model_path)
    assert model_path.exists()

    # Load
    new_calibrator = UncertaintyCalibrator()
    new_calibrator.load(model_path)
    assert new_calibrator._is_trained
    assert new_calibrator.backend == "simple"

    # Predictions should match
    test_features = features[:5]
    pred1 = calibrator.predict(test_features)
    pred2 = new_calibrator.predict(test_features)
    np.testing.assert_allclose(pred1, pred2, rtol=1e-5)


def test_extract_features_from_ground_point():
    """Test feature extraction from ground point dict."""
    point = {
        "x": 100.0,
        "y": 200.0,
        "z": 50.0,
        "method": "opensfm",
        "uncertainty": 0.25,
        "sem_prob": 0.85,
        "tri_angle_deg": 12.5,
        "support": 3,
    }

    camera_centers = [
        np.array([95.0, 195.0, 51.5]),
        np.array([105.0, 205.0, 51.8]),
        np.array([100.0, 190.0, 52.0]),
    ]

    mask_values = [0.9, 0.85, 0.8]

    local_points = [
        np.array([100.5, 200.5, 50.1]),
        np.array([99.5, 199.5, 49.9]),
        np.array([101.0, 201.0, 50.2]),
    ]

    feat = extract_features_from_ground_point(
        point, camera_centers, mask_values, local_points
    )

    assert feat.tri_angle_deg == 12.5
    assert feat.view_count == 3
    assert feat.sem_prob == 0.85
    assert feat.base_uncertainty == 0.25
    assert feat.min_distance > 0.0
    assert feat.max_baseline > 0.0
    assert feat.mask_variance >= 0.0
    assert feat.local_density > 0.0
    assert feat.method == "opensfm"


def test_prepare_training_data_from_consensus():
    """Test training data preparation from consensus points."""
    consensus_points = [
        {
            "x": 100.0,
            "y": 200.0,
            "z": 50.0,
            "tri_angle_deg": 15.0,
            "support": 3,
            "sem_prob": 0.85,
            "uncertainty": 0.25,
            "method": "consensus",
        },
        {
            "x": 105.0,
            "y": 205.0,
            "z": 51.0,
            "tri_angle_deg": 10.0,
            "support": 2,
            "sem_prob": 0.75,
            "uncertainty": 0.30,
            "method": "consensus",
        },
    ]

    source_points = {
        "opensfm": [
            {"x": 100.0, "y": 200.0, "z": 50.1},
            {"x": 105.0, "y": 205.0, "z": 51.05},
        ],
        "colmap": [
            {"x": 100.0, "y": 200.0, "z": 49.9},
            {"x": 105.0, "y": 205.0, "z": 50.95},
        ],
        "vo": [
            {"x": 100.0, "y": 200.0, "z": 50.05},
        ],
    }

    features, errors = prepare_training_data_from_consensus(
        consensus_points, source_points
    )

    assert len(features) == len(errors)
    assert len(features) >= 1  # At least one valid training sample

    for feat, err in zip(features, errors):
        assert isinstance(feat, UncertaintyFeatures)
        assert 0.0 <= err < 1.0  # Reasonable error range


def test_calibrator_untrained_prediction():
    """Test that untrained model falls back to base uncertainties."""
    calibrator = UncertaintyCalibrator()

    features = [
        UncertaintyFeatures(
            tri_angle_deg=10.0,
            view_count=3,
            sem_prob=0.8,
            base_uncertainty=0.25,
            min_distance=8.0,
            max_baseline=4.0,
            mask_variance=0.04,
            local_density=2.0,
            method="opensfm",
        ),
        UncertaintyFeatures(
            tri_angle_deg=5.0,
            view_count=2,
            sem_prob=0.7,
            base_uncertainty=0.35,
            min_distance=12.0,
            max_baseline=3.0,
            mask_variance=0.06,
            local_density=1.5,
            method="colmap",
        ),
    ]

    predictions = calibrator.predict(features)

    # Should return base uncertainties
    assert predictions[0] == 0.25
    assert predictions[1] == 0.35


def test_calibrator_feature_scaling():
    """Test feature normalization."""
    calibrator = UncertaintyCalibrator(backend="simple")

    # Create training data with different scales
    rng = np.random.default_rng(777)
    features = []
    errors = []

    for _ in range(100):
        feat = UncertaintyFeatures(
            tri_angle_deg=float(rng.uniform(1, 30)),
            view_count=int(rng.integers(2, 6)),
            sem_prob=float(rng.uniform(0.6, 0.95)),
            base_uncertainty=float(rng.uniform(0.1, 0.5)),
            min_distance=float(rng.uniform(5, 20)),
            max_baseline=float(rng.uniform(1, 10)),
            mask_variance=float(rng.uniform(0.01, 0.15)),
            local_density=float(rng.uniform(0.5, 5.0)),
            method="opensfm",
        )
        features.append(feat)
        errors.append(0.2 + rng.normal(0, 0.05))

    calibrator.train(features, errors)

    # Check scaler exists
    assert calibrator.scaler is not None
    assert "mean" in calibrator.scaler
    assert "std" in calibrator.scaler

    # Predictions should be reasonable despite scale differences
    pred = calibrator.predict(features[:5])
    assert np.all(pred >= 0.05)
    assert np.all(pred <= 0.6)
