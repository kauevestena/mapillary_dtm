from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from dtm_from_mapillary.cli.pipeline import (
    _source_dtms_on_grid,
    _strict_preflight,
)
from dtm_from_mapillary.common_core import FrameMeta, GroundPoint
from dtm_from_mapillary.depth import monodepth
from dtm_from_mapillary.fusion.heightmap_fusion import GridSpec
from dtm_from_mapillary.qa.qa_external import compare_to_geotiff, main as qa_external_main
from dtm_from_mapillary.semantics import ground_masks
from scripts.setup_production_models import main as setup_models_main


def _frame(image_id: str = "img-1", seq_id: str = "seq-1") -> FrameMeta:
    return FrameMeta(
        image_id=image_id,
        seq_id=seq_id,
        captured_at_ms=0,
        lon=-48.0,
        lat=-27.0,
        alt_ellip=10.0,
        camera_type="perspective",
        cam_params={"width": 4, "height": 4},
        quality_score=0.9,
    )


def _write_tif(path: Path, arr: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=Affine.translation(100.0, 200.0) * Affine.scale(1.0, -1.0),
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


def test_external_qa_writes_residual_rasters_and_masks_reference_zero(tmp_path: Path) -> None:
    generated = tmp_path / "generated.tif"
    reference = tmp_path / "reference.tif"
    out_dir = tmp_path / "qa"
    gen = np.full((4, 4), 12.0, dtype=np.float32)
    ref = np.full((4, 4), 10.0, dtype=np.float32)
    ref[1, 1] = 0.0
    _write_tif(generated, gen)
    _write_tif(reference, ref)

    stats = compare_to_geotiff(
        str(generated),
        str(reference),
        out_dir=out_dir,
        reference_nodata_values=[0.0],
    )

    assert stats["n"] == 15
    assert stats["rmse_z"] == pytest.approx(2.0)
    assert stats["p95_abs_z"] == pytest.approx(2.0)
    for name in ("dz.tif", "abs_dz.tif", "slope_diff_deg.tif", "external_qa_summary.json"):
        assert (out_dir / name).exists()


def test_external_qa_cli(tmp_path: Path) -> None:
    generated = tmp_path / "generated.tif"
    reference = tmp_path / "reference.tif"
    out_dir = tmp_path / "qa_cli"
    _write_tif(generated, np.ones((3, 3), dtype=np.float32))
    _write_tif(reference, np.ones((3, 3), dtype=np.float32))

    code = qa_external_main(
        [
            "--generated-dtm",
            str(generated),
            "--reference-dtm",
            str(reference),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    assert (out_dir / "external_qa_summary.json").exists()


def test_strict_ground_masks_reject_unprovenanced_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    np.savez_compressed(tmp_path / "img-1.npz", prob=np.ones((4, 4), dtype=np.float32))
    monkeypatch.setattr(ground_masks, "_init_model_masker", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="Ground mask missing"):
        ground_masks.prepare({"seq-1": [frame]}, out_dir=tmp_path)


def test_strict_depth_rejects_unprovenanced_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _frame()
    np.savez_compressed(
        tmp_path / "img-1.npz",
        depth=np.ones((4, 4), dtype=np.float32),
        uncertainty=np.ones((4, 4), dtype=np.float32) * 0.2,
    )
    monkeypatch.setattr(monodepth, "_init_default_adapter", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="Monodepth prediction unavailable"):
        monodepth.predict_depths({"seq-1": [frame]}, out_dir=tmp_path)


def test_source_dtms_are_rasterized_on_fused_grid() -> None:
    gp = GroundPoint(
        x=0.25,
        y=0.25,
        z=5.0,
        method="test",
        seq_id="seq-1",
        image_ids=["img-1", "img-2"],
        view_count=2,
        sem_prob=0.9,
        tri_angle_deg=4.0,
        uncertainty_m=0.2,
    )
    dtms = _source_dtms_on_grid({"opensfm": [gp], "colmap": []}, GridSpec(0, 0, 2, 2, 0.5))

    assert np.isfinite(dtms["opensfm"][0, 0])
    assert np.isnan(dtms["opensfm"][0, 1])
    assert np.isnan(dtms["colmap"]).all()


def test_strict_preflight_reports_missing_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    imagery = tmp_path / "imagery" / "seq-1"
    imagery.mkdir(parents=True)
    image_path = imagery / "img-1.jpg"
    image_path.write_bytes(b"not a real image")
    for key in (
        "GROUND_MASK_MODEL_PATH",
        "MONODEPTH_MODEL_PATH",
        "GROUND_MASK_MODEL_ID",
        "MONODEPTH_MODEL_ID",
        "OPEN_SFM_FORCE_SYNTHETIC",
        "COLMAP_FORCE_SYNTHETIC",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("dtm_from_mapillary.cli.pipeline._hf_model_cached", lambda model_id: False)
    monkeypatch.setattr("dtm_from_mapillary.cli.pipeline._docker_image_available", lambda image: True)
    monkeypatch.setattr("dtm_from_mapillary.cli.pipeline._sample_readable_errors", lambda paths: [])

    with pytest.raises(RuntimeError, match="ground mask model"):
        _strict_preflight(
            {"seq-1": [_frame()]},
            imagery_root_path=tmp_path / "imagery",
            reference_dtm=None,
        )


def test_setup_production_models_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_snapshot_download(repo_id, revision, cache_dir, local_files_only):
        snapshot = Path(cache_dir) / repo_id.replace("/", "--") / "snapshots" / "abc123"
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "config.json").write_text(json.dumps({"repo_id": repo_id}), encoding="utf8")
        return str(snapshot)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    manifest = tmp_path / "production_models.json"

    code = setup_models_main(
        [
            "--accept-model-licenses",
            "--cache-dir",
            str(tmp_path / "hf"),
            "--manifest-out",
            str(manifest),
        ]
    )

    assert code == 0
    payload = json.loads(manifest.read_text(encoding="utf8"))
    assert payload["accepted_model_licenses"] is True
    assert {item["name"] for item in payload["models"]} == {"ground_segmentation", "monodepth"}
