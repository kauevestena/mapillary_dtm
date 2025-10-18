"""Integration tests for consensus voting using fixture datasets."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if "dtm_from_mapillary" not in sys.modules:
    pkg = types.ModuleType("dtm_from_mapillary")
    pkg.__path__ = [str(ROOT)]
    sys.modules["dtm_from_mapillary"] = pkg
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dtm_from_mapillary.common_core import GroundPoint
from dtm_from_mapillary.ground.recon_consensus import agree
from dtm_from_mapillary import constants


def load_ground_points_from_json(path: Path) -> list[GroundPoint]:
    """Load ground points from JSON fixture file."""
    data = json.loads(path.read_text())
    points = []
    for item in data:
        points.append(
            GroundPoint(
                x=item["x"],
                y=item["y"],
                z=item["z"],
                method=item.get("source", "test"),
                seq_id="test-seq",
                image_ids=["img-1"],
                view_count=1,
                sem_prob=item.get("sem_prob", 0.8),
                uncertainty_m=item.get("uncertainty_m", 0.1),
                tri_angle_deg=item.get("tri_angle_deg"),
            )
        )
    return points


def test_consensus_with_fixture_data():
    """Test consensus voting with captured fixture datasets."""
    fixture_dir = ROOT / "qa" / "data" / "consensus_fixture"
    
    # Load fixture data
    points_A = load_ground_points_from_json(fixture_dir / "ground_points_A.json")
    points_B = load_ground_points_from_json(fixture_dir / "ground_points_B.json")
    points_C = load_ground_points_from_json(fixture_dir / "ground_points_C.json")
    
    # Run consensus
    consensus = agree(points_A, points_B, points_C)
    
    # Should find consensus at locations where 2+ sources agree
    assert len(consensus) > 0
    
    # Check that all consensus points have required fields
    for cp in consensus:
        assert "x" in cp
        assert "y" in cp
        assert "z" in cp
        assert "sources" in cp
        assert "support" in cp
        assert "sem_prob" in cp
        assert "uncertainty" in cp
        
        # Should have at least 2 sources
        assert len(cp["sources"]) >= 2
        assert cp["support"] >= 2
        
        # Values should be reasonable
        assert 0.0 <= cp["sem_prob"] <= 1.0
        assert 0.05 <= cp["uncertainty"] <= 0.6


def test_consensus_insufficient_overlap():
    """Test consensus when sources have insufficient spatial overlap."""
    # Create points that don't overlap spatially
    points_A = [
        GroundPoint(x=100.0, y=100.0, z=10.0, method="A", seq_id="s1", image_ids=["i1"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
        GroundPoint(x=101.0, y=100.0, z=10.0, method="A", seq_id="s1", image_ids=["i1"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    points_B = [
        GroundPoint(x=200.0, y=200.0, z=15.0, method="B", seq_id="s1", image_ids=["i2"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
        GroundPoint(x=201.0, y=200.0, z=15.0, method="B", seq_id="s1", image_ids=["i2"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    points_C = [
        GroundPoint(x=300.0, y=300.0, z=20.0, method="C", seq_id="s1", image_ids=["i3"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
        GroundPoint(x=301.0, y=300.0, z=20.0, method="C", seq_id="s1", image_ids=["i3"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    # Should return no consensus points due to lack of overlap
    consensus = agree(points_A, points_B, points_C)
    assert len(consensus) == 0


def test_consensus_height_disagreement():
    """Test consensus when sources disagree on height."""
    grid_res = constants.GRID_RES_M
    
    # Create points at same location but different heights (beyond threshold)
    points_A = [
        GroundPoint(x=100.0, y=100.0, z=10.0, method="A", seq_id="s1", image_ids=["i1"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    points_B = [
        GroundPoint(x=100.0, y=100.0, z=15.0, method="B", seq_id="s1", image_ids=["i2"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    points_C = [
        GroundPoint(x=100.0, y=100.0, z=20.0, method="C", seq_id="s1", image_ids=["i3"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    # Should return no consensus due to height disagreement
    consensus = agree(points_A, points_B, points_C)
    assert len(consensus) == 0


def test_consensus_single_source():
    """Test that consensus requires multiple sources."""
    # Only provide points from one source
    points_A = [
        GroundPoint(x=100.0, y=100.0, z=10.0, method="A", seq_id="s1", image_ids=["i1"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
        GroundPoint(x=101.0, y=100.0, z=10.0, method="A", seq_id="s1", image_ids=["i1"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    # Should return no consensus points (need 2+ sources)
    consensus = agree(points_A, None, None)
    assert len(consensus) == 0


def test_consensus_height_agreement():
    """Test consensus when sources agree within threshold."""
    # Create points that agree within DZ_MAX_M threshold
    dz_max = constants.DZ_MAX_M
    
    points_A = [
        GroundPoint(x=100.0, y=100.0, z=10.0, method="A", seq_id="s1", image_ids=["i1"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    points_B = [
        GroundPoint(x=100.0, y=100.0, z=10.0 + dz_max * 0.5, method="B", seq_id="s1", image_ids=["i2"], view_count=1, sem_prob=0.85, uncertainty_m=0.15, tri_angle_deg=45.0),
    ]
    
    # Should find consensus
    consensus = agree(points_A, points_B, None)
    assert len(consensus) == 1
    assert len(consensus[0]["sources"]) == 2


def test_consensus_weighted_averaging():
    """Test that consensus uses uncertainty-weighted averaging."""
    # Create points with different uncertainties
    points_A = [
        GroundPoint(x=100.0, y=100.0, z=10.0, method="A", seq_id="s1", image_ids=["i1"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    points_B = [
        GroundPoint(x=100.0, y=100.0, z=10.2, method="B", seq_id="s1", image_ids=["i2"], view_count=1, sem_prob=0.85, uncertainty_m=0.5, tri_angle_deg=45.0),
    ]
    
    consensus = agree(points_A, points_B, None)
    assert len(consensus) == 1
    
    # More certain point (A) should have more weight
    # So consensus uncertainty should be closer to 0.1 than 0.5
    assert consensus[0]["uncertainty"] < 0.3


def test_consensus_lower_envelope():
    """Test that consensus uses lower envelope for height."""
    # Create points at same location with height variation
    points_A = [
        GroundPoint(x=100.0, y=100.0, z=10.0, method="A", seq_id="s1", image_ids=["i1"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
        GroundPoint(x=100.0, y=100.0, z=10.5, method="A", seq_id="s1", image_ids=["i1"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    points_B = [
        GroundPoint(x=100.0, y=100.0, z=10.2, method="B", seq_id="s1", image_ids=["i2"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
        GroundPoint(x=100.0, y=100.0, z=10.4, method="B", seq_id="s1", image_ids=["i2"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    consensus = agree(points_A, points_B, None)
    assert len(consensus) == 1
    
    # Should use lower envelope (percentile)
    # Exact value depends on LOWER_ENVELOPE_Q but should be closer to min
    assert consensus[0]["z"] <= 10.3


def test_consensus_empty_inputs():
    """Test consensus with empty or None inputs."""
    consensus = agree(None, None, None)
    assert len(consensus) == 0
    
    consensus = agree([], [], [])
    assert len(consensus) == 0


def test_consensus_custom_grid_resolution():
    """Test consensus with custom grid resolution."""
    # Create points that would be in different cells with small grid
    # but same cell with larger grid
    points_A = [
        GroundPoint(x=100.0, y=100.0, z=10.0, method="A", seq_id="s1", image_ids=["i1"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    points_B = [
        GroundPoint(x=100.3, y=100.3, z=10.0, method="B", seq_id="s1", image_ids=["i2"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    # With default grid (0.5m), should be in same cell
    consensus = agree(points_A, points_B, None)
    assert len(consensus) == 1
    
    # With very fine grid (0.1m), should be in different cells
    consensus = agree(points_A, points_B, None, grid_res=0.1)
    assert len(consensus) == 0


def test_consensus_custom_height_threshold():
    """Test consensus with custom height threshold."""
    points_A = [
        GroundPoint(x=100.0, y=100.0, z=10.0, method="A", seq_id="s1", image_ids=["i1"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    points_B = [
        GroundPoint(x=100.0, y=100.0, z=10.3, method="B", seq_id="s1", image_ids=["i2"], view_count=1, sem_prob=0.9, uncertainty_m=0.1, tri_angle_deg=45.0),
    ]
    
    # With tight threshold, should not agree
    consensus = agree(points_A, points_B, None, dz_max=0.1)
    assert len(consensus) == 0
    
    # With loose threshold, should agree
    consensus = agree(points_A, points_B, None, dz_max=0.5)
    assert len(consensus) == 1
