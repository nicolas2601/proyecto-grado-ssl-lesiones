from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "scripts/release_v10"


def test_runtime_image_has_pinned_base_and_non_root_offline_entrypoint():
    text = (ROOT / "Dockerfile").read_text()
    assert "FROM python@sha256:" in text
    assert "USER 65532:65532" in text
    assert '"/app/runner.py"' in text
    assert '"onnx"' in text
    assert "torch.hub" not in text
    assert "COPY bundle/ /app/" in text


def test_onnx_runtime_dependencies_are_explicit_and_do_not_require_torch():
    text = (ROOT / "requirements-onnx.txt").read_text()
    assert "numpy==" in text and "onnxruntime==" in text and "Pillow==" in text
    assert "torch==" not in text
