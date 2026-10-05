"""The browser preprocessing (docs/release/web/preprocess.js) equals preprocess.py.

Runs the JavaScript under node on the same RGB pixels. Skips when node is
absent (it is present on GitHub's ubuntu runners). Sizes include odd crop
margins, where Python's half-to-even round() differs from Math.round.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dermssl.release import preprocess as pp

JS = Path(__file__).resolve().parents[1] / "docs" / "release" / "web" / "preprocess.js"
SIZES = [(600, 450), (450, 600), (257, 300), (1024, 767), (255, 255), (333, 481)]
RUNNER = """
import { readFileSync, writeFileSync } from "node:fs";
import { preprocessRGB } from %s;
const [w, h] = [Number(process.argv[2]), Number(process.argv[3])];
const out = preprocessRGB(new Uint8Array(readFileSync(process.argv[4])), w, h);
writeFileSync(process.argv[5], Buffer.from(out.buffer));
"""


@pytest.mark.parametrize("size", SIZES)
def test_javascript_matches_python(size, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    width, height = size
    rgb = np.random.default_rng(width + height).integers(0, 256, (height, width, 3), dtype=np.uint8)
    (tmp_path / "in.rgb").write_bytes(rgb.tobytes())
    runner = tmp_path / "run.mjs"
    runner.write_text(RUNNER % repr(JS.as_uri()))
    subprocess.run([node, str(runner), str(width), str(height), str(tmp_path / "in.rgb"),
                    str(tmp_path / "out.f32")], check=True, timeout=120)
    got = np.frombuffer((tmp_path / "out.f32").read_bytes(), dtype=np.float32).reshape(3, 224, 224)
    np.testing.assert_allclose(got, pp.preprocess_image(Image.fromarray(rgb)), rtol=0, atol=1e-6)
