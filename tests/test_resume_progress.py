from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from dtm_from_mapillary.cli.pipeline import (
    RUN_STAGES,
    _RunState,
    _invalidate_forced_stages,
    _validate_depth_cache,
    _validate_mask_cache,
)
from dtm_from_mapillary.common_core import FrameMeta
from dtm_from_mapillary.depth import monodepth
from dtm_from_mapillary.semantics import ground_masks


def _frame(image_id: str = "img-1", seq_id: str = "seq-1") -> FrameMeta:
    return FrameMeta(
        image_id=image_id,
        seq_id=seq_id,
        captured_at_ms=0,
        lon=-48.0,
        lat=-27.0,
        alt_ellip=10.0,
        camera_type="perspective",
        cam_params={"width": 8, "height": 6},
        quality_score=0.9,
    )


class _FakeProgress:
    def __init__(self) -> None:
        self.total = 0
        self.desc = ""
        self.updates = 0
        self.postfixes: list[dict[str, int]] = []

    def __enter__(self) -> "_FakeProgress":
        return self

    def __exit__(self, *args) -> None:
        return None

    def update(self, value: int = 1) -> None:
        self.updates += value

    def set_postfix(self, **kwargs) -> None:
        self.postfixes.append(dict(kwargs))


def test_run_state_persists_stage_status(tmp_path: Path) -> None:
    path = tmp_path / "run_state.json"
    state = _RunState(path, resume=True)
    state.start("masks", inputs={"cache_dir": "cache/masks"})
    state.complete("masks", outputs={"cache_dir": "cache/masks"}, counts={"valid": 1})

    reloaded = _RunState(path, resume=True)

    assert reloaded.is_complete("masks")
    assert reloaded.stage("masks")["outputs"]["cache_dir"] == "cache/masks"
    assert reloaded.stage("masks")["counts"]["valid"] == 1


def test_force_stage_invalidates_downstream_stages() -> None:
    payload = {"stages": {stage: {"status": "complete"} for stage in RUN_STAGES}}

    invalidated = _invalidate_forced_stages(payload, ["colmap"])

    assert invalidated[0] == "colmap"
    assert payload["stages"]["opensfm"]["status"] == "complete"
    for stage in RUN_STAGES[RUN_STAGES.index("colmap") :]:
        assert payload["stages"][stage]["status"] == "pending"
        assert payload["stages"][stage]["reason"] == "forced from colmap"


def test_strict_cache_validators_reject_unprovenanced_or_synthetic(tmp_path: Path) -> None:
    frame = _frame()
    seqs = {"seq-1": [frame]}
    mask_dir = tmp_path / "masks"
    depth_dir = tmp_path / "depth"
    mask_dir.mkdir()
    depth_dir.mkdir()
    np.savez_compressed(mask_dir / "img-1.npz", prob=np.ones((2, 2), dtype=np.float32))
    np.savez_compressed(
        depth_dir / "img-1.npz",
        depth=np.ones((2, 2), dtype=np.float32),
        uncertainty=np.ones((2, 2), dtype=np.float32),
        source_type="synthetic",
    )

    assert not _validate_mask_cache(seqs, mask_dir, strict=True)
    assert not _validate_depth_cache(seqs, depth_dir, strict=True)
    assert _validate_depth_cache(seqs, depth_dir, strict=False)


def test_ground_masks_cached_resume_does_not_initialize_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _frame()
    np.savez_compressed(
        tmp_path / "img-1.npz",
        prob=np.ones((2, 2), dtype=np.float32),
        source_type="model",
        backend="test",
        model_id="model",
    )
    fake = _FakeProgress()
    monkeypatch.setattr(
        ground_masks,
        "_progress_bar",
        lambda total, desc, enabled: fake,
    )
    monkeypatch.setattr(
        ground_masks,
        "_init_model_masker",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("model initialized")),
    )

    result = ground_masks.prepare(
        {"seq-1": [frame]},
        out_dir=tmp_path,
        progress=True,
    )

    assert result == {"seq-1": [tmp_path / "img-1.npz"]}
    assert fake.updates == 1
    assert fake.postfixes[-1] == {"cached": 1, "generated": 0}


def test_monodepth_cached_resume_does_not_initialize_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _frame()
    np.savez_compressed(
        tmp_path / "img-1.npz",
        depth=np.ones((2, 2), dtype=np.float32),
        uncertainty=np.ones((2, 2), dtype=np.float32) * 0.2,
        source_type="model",
        backend="test",
        model_id="model",
    )
    fake = _FakeProgress()
    monkeypatch.setattr(
        monodepth,
        "_progress_bar",
        lambda total, desc, enabled: fake,
    )
    monkeypatch.setattr(
        monodepth,
        "_init_default_adapter",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("adapter initialized")),
    )

    result = monodepth.predict_depths(
        {"seq-1": [frame]},
        out_dir=tmp_path,
        progress=True,
    )

    assert result["seq-1"]["img-1"]["depth"].shape == (2, 2)
    assert fake.updates == 1
    assert fake.postfixes[-1] == {"cached": 1, "generated": 0}


def test_cli_help_exposes_resume_progress_flags() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dtm_from_mapillary.cli.pipeline", "run", "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    assert "--resume" in result.stdout
    assert "--no-resume" in result.stdout
    assert "--force-stage" in result.stdout
    assert "--progress" in result.stdout
    assert "--no-progress" in result.stdout
