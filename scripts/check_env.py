#!/usr/bin/env python3
"""Environment validation helper for the Mapillary DTM runtime."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VENV_BIN = _REPO_ROOT / ".venv" / "bin"
if _VENV_BIN.exists():
    _path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{_VENV_BIN}{os.pathsep}{_path}"


Result = Tuple[bool, str]


@dataclass(frozen=True)
class Check:
    name: str
    required: bool
    runner: Callable[[argparse.Namespace], Result]


def _format_status(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def check_python(_: argparse.Namespace) -> Result:
    target = (3, 11)
    version = sys.version_info
    if version < target:
        return False, f"Python {target[0]}.{target[1]}+ required, detected {version.major}.{version.minor}"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"


def check_module(module: str, hint: str | None = None) -> Callable[[argparse.Namespace], Result]:
    def _runner(_: argparse.Namespace) -> Result:
        try:
            importlib.import_module(module)
            return True, "import ok"
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            message = f"import failed: {exc}"
            if hint:
                message += f" ({hint})"
            return False, message

    return _runner


def check_command(command: str, hint: str | None = None, run_args: Iterable[str] | None = None) -> Callable[[argparse.Namespace], Result]:
    def _runner(_: argparse.Namespace) -> Result:
        binary = shutil.which(command)
        if not binary:
            msg = "not on PATH"
            if hint:
                msg += f" ({hint})"
            return False, msg
        if run_args:
            try:
                subprocess.run([binary, *run_args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=10)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
                return False, f"exec failed: {exc}"
        return True, f"found at {binary}"

    return _runner


DEFAULT_OPENSFM_IMAGE = "freakthemighty/opensfm:latest"


def check_docker_opensfm(args: argparse.Namespace) -> Result:
    if not shutil.which("docker"):
        return False, "docker not detected"
    image = getattr(args, "opensfm_image", DEFAULT_OPENSFM_IMAGE)
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=10,
        )
        return True, f"docker image {image} available"
    except subprocess.CalledProcessError:
        return False, f"docker image {image} missing (run `docker pull {image}`)"
    except subprocess.TimeoutExpired as exc:
        return False, f"docker image inspect timeout: {exc}"


def check_cuda(_: argparse.Namespace) -> Result:
    nvcc = shutil.which("nvcc")
    smi = shutil.which("nvidia-smi")
    if nvcc or smi:
        locations = [path for path in (nvcc, smi) if path]
        return True, "CUDA tooling detected: " + ", ".join(locations)
    return False, "CUDA toolkit / driver not found (required when enabling GPU paths)"


def check_env_var(name: str, required: bool) -> Callable[[argparse.Namespace], Result]:
    def _runner(_: argparse.Namespace) -> Result:
        value = os.getenv(name)
        if value:
            return True, "set"
        status = "required" if required else "recommended"
        return (not required), f"{status} env var not set"

    return _runner


_TOKEN_FILE_CANDIDATES = (
    "mapillary_token",
    ".secrets/mapillary_token",
    "config/mapillary_token",
)
_ENV_FILE_CANDIDATES = (".env", "config/runtime.env")


def _parse_token_line(line: str) -> Optional[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" in stripped:
        key, value = stripped.split("=", 1)
        if key.strip() != "MAPILLARY_TOKEN":
            return None
        candidate = value.strip().strip('"').strip("'")
    else:
        candidate = stripped
    return candidate or None


def _read_token_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf8").splitlines():
            candidate = _parse_token_line(line)
            if candidate:
                return candidate
    except OSError:
        return None
    return None


def resolve_mapillary_token() -> Tuple[Optional[str], str]:
    """Discover Mapillary token from env or known files (no secret logging)."""
    env_token = os.getenv("MAPILLARY_TOKEN")
    if env_token:
        return env_token.strip(), "env MAPILLARY_TOKEN"

    token_file_env = os.getenv("MAPILLARY_TOKEN_FILE")
    if token_file_env:
        token = _read_token_file(Path(token_file_env).expanduser())
        if token:
            return token, f"MAPILLARY_TOKEN_FILE={token_file_env}"

    repo_root = Path(__file__).resolve().parent.parent
    for rel_path in _ENV_FILE_CANDIDATES:
        token = _read_token_file(repo_root / rel_path)
        if token:
            return token, rel_path

    for rel_path in _TOKEN_FILE_CANDIDATES:
        token = _read_token_file(repo_root / rel_path)
        if token:
            return token, rel_path

    return None, "missing"


def check_mapillary_token(_: argparse.Namespace) -> Result:
    token, source = resolve_mapillary_token()
    if token:
        return True, f"found in {source}"
    return False, "required (set MAPILLARY_TOKEN or create mapillary_token file)"


def base_checks() -> List[Check]:
    return [
        Check("python", True, check_python),
        Check("numpy", True, check_module("numpy")),
        Check("rasterio", True, check_module("rasterio")),
        Check("geopandas", True, check_module("geopandas")),
        Check("osmnx", True, check_module("osmnx")),
        Check("triangle", True, check_module("triangle", hint="install system build tools")),
        Check("laspy[lazrs]", False, check_module("laspy")),
        Check("colmap", True, check_command("colmap", hint="install COLMAP and expose on PATH", run_args=["--help"])),
        Check("docker", True, check_command("docker", hint="install Docker Engine")),
        Check("mapillary token", True, check_mapillary_token),
    ]


def optional_checks() -> List[Check]:
    return [
        Check("opensfm docker image", False, check_docker_opensfm),
        Check("torch", False, check_module("torch", hint="install from requirements-optional.txt")),
        Check("torchvision", False, check_module("torchvision", hint="install from requirements-optional.txt")),
        Check("CUDA toolchain", False, check_cuda),
        Check("nvidia-smi", False, check_command("nvidia-smi", hint="install NVIDIA driver")),
    ]


def run_checks(checks: Iterable[Check], args: argparse.Namespace) -> Tuple[List[dict], bool]:
    results: List[dict] = []
    all_ok = True
    for check in checks:
        ok, message = check.runner(args)
        results.append(
            {
                "name": check.name,
                "status": "pass" if ok else "fail",
                "required": check.required,
                "message": message,
            }
        )
        if check.required and not ok:
            all_ok = False
    return results, all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Mapillary DTM runtime environment.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run optional / GPU checks in addition to required ones.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable table.",
    )
    parser.add_argument(
        "--opensfm-image",
        default=DEFAULT_OPENSFM_IMAGE,
        help="Docker image tag to validate for OpenSfM (default: %(default)s).",
    )
    return parser.parse_args()


def render_table(rows: List[dict]) -> None:
    width_name = max(len(row["name"]) for row in rows) + 2
    print(f"{'Check':{width_name}}Status  Message")
    print("-" * (width_name + 40))
    for row in rows:
        status = _format_status(row["status"] == "pass")
        print(f"{row['name']:{width_name}}{status:<6} {row['message']}")


def main() -> int:
    args = parse_args()
    rows, required_ok = run_checks(base_checks(), args)
    if args.full:
        optional_rows, _ = run_checks(optional_checks(), args)
        rows.extend(optional_rows)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        render_table(rows)
    return 0 if required_ok else 1


if __name__ == "__main__":
    sys.exit(main())
