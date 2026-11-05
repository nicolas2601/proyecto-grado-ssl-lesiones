"""Create a fully isolated public-vision environment without modifying legacy packages."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "environments/public-encoders-20261025-v1"


def main():
    FAMILY.parent.mkdir(exist_ok=True)
    if not FAMILY.parent.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Environment parent escapes isolated project")
    FAMILY.mkdir(exist_ok=False)
    started = time.monotonic()
    clean = {key: value for key, value in os.environ.items()
             if not key.startswith("PIP_") and not key.startswith("PYTHON")}
    clean.update(PIP_CONFIG_FILE=os.devnull, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                 HF_HOME=str(FAMILY / "hf-cache"), TORCH_HOME=str(FAMILY / "torch-cache"),
                 XDG_CACHE_HOME=str(FAMILY / "cache"), HF_HUB_DISABLE_TELEMETRY="1",
                 HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    status = {"status": "PREPARING", "environment_root": str(FAMILY),
              "legacy_environments_modified": False, "source": "PyPI wheels only"}
    try:
        package_metadata = {}
        for name, version in (("torch", "2.9.1"), ("torchvision", "0.24.1")):
            with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
                raw = response.read(5_000_001)
            if len(raw) > 5_000_000:
                raise ValueError("Unexpected package metadata size")
            (FAMILY / f"pypi-{name}-{version}.json").write_bytes(raw)
            package_metadata[name] = json.loads(raw)
        if not any("torch==2.9.1" in value.replace(" ", "").replace("(", "").replace(")", "")
                   for value in package_metadata["torchvision"]["info"]["requires_dist"]):
            raise ValueError("Published TorchVision/Torch versions are not coupled as expected")
        if not any("cuda-runtime-cu12" in value and "12.8" in value
                   for value in package_metadata["torch"]["info"]["requires_dist"]):
            raise ValueError("Expected CUDA12.8 Torch wheels for the current Blackwell driver")
        venv = FAMILY / "venv"
        subprocess.run([sys.executable, "-I", "-m", "venv", str(venv)], check=True, env=clean)
        python = str(venv / "bin/python")
        check = "import pathlib,sys; assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(" + repr(str(venv)) + ").resolve(); assert not any('pg_night_venv' in p or 'pg_clean_inference' in p for p in sys.path); print(sys.version)"
        subprocess.run([python, "-I", "-c", check], check=True, env=clean)
        pip = [python, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
               "--index-url", "https://pypi.org/simple", "--cache-dir", str(FAMILY / "pip-cache"),
               "--only-binary=:all:"]
        direct = ["torch==2.9.1", "torchvision==0.24.1", "transformers", "huggingface_hub", "pillow", "safetensors", "numpy"]
        report = FAMILY / "resolved-pypi-wheels.json"
        subprocess.run(pip + ["--dry-run", "--report", str(report)] + direct, check=True, env=clean)
        resolution = json.loads(report.read_text())
        lines = []
        for package in resolution["install"]:
            sha = package["download_info"]["archive_info"]["hashes"]["sha256"]
            name, version = package["metadata"]["name"], package["metadata"]["version"]
            lines.append(f"{name}=={version} --hash=sha256:{sha}")
        lock = FAMILY / "requirements-pinned.txt"
        lock.write_text("\n".join(lines) + "\n")
        subprocess.run(pip + ["--require-hashes", "-r", str(lock)], check=True, env=clean)
        subprocess.run([python, "-I", "-m", "pip", "check"], check=True, env=clean)
        validate = "import json,pathlib,sys,torch,torchvision,transformers,huggingface_hub; from transformers import Dinov2Model,SiglipVisionModel; assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(" + repr(str(venv)) + ").resolve(); assert not any('pg_night_venv' in p or 'pg_clean_inference' in p for p in sys.path); print(json.dumps({'python':sys.version,'torch':torch.__version__,'CUDA_runtime':torch.version.cuda,'torchvision':torchvision.__version__,'transformers':transformers.__version__,'huggingface_hub':huggingface_hub.__version__,'vision_classes':[Dinov2Model.__name__,SiglipVisionModel.__name__]}))"
        versions = subprocess.check_output([python, "-I", "-c", validate], text=True, env=clean)
        (FAMILY / "verified-versions.json").write_text(versions)
        status.update(status="READY_FOR_GPU_SMOKE", python=python,
                      requirement_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
                      versions=json.loads(versions), package_count=len(lines),
                      inherited_system_site_packages=False, GPU_compute_used=False)
    except Exception as error:
        status.update(status="FAILED", error_type=type(error).__name__,
                      error=str(error)[:500], GPU_compute_used=False)
        raise
    finally:
        status["seconds"] = time.monotonic() - started
        (FAMILY / "setup-receipt.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
