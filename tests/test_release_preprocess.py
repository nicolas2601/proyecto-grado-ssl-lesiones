"""The release preprocessing must be the evaluation transform, without torch.

The evaluated features came from ``cluster_frozen_bank.eval_transform``:
Resize(255, BILINEAR) on the shorter side, CenterCrop(224), ToTensor and
ImageNet normalisation. The numpy/PIL copy shipped in the release is compared
against that exact torchvision pipeline, not against a description of it.
"""

import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from dermssl.release import preprocess as pp

SIZES = [(600, 450), (450, 600), (601, 451), (300, 260), (255, 255), (1024, 767)]


def _image(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    base[..., 0] = (base[..., 0] // 4 + (xx * 191 // width)).astype(np.uint8)  # structure, not only noise
    return Image.fromarray(base)


@pytest.mark.parametrize("size", SIZES)
def test_matches_the_torchvision_eval_transform(size):
    pytest.importorskip("torchvision")
    from scripts import cluster_frozen_bank as bank

    image = _image(*size, seed=size[0] * 7 + size[1])
    expected = bank.eval_transform(224)(image).numpy()
    got = pp.preprocess_image(image)
    assert got.dtype == np.float32 and got.shape == (3, 224, 224)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_resize_keeps_the_aspect_ratio_like_torchvision():
    assert pp.resized_size(600, 450) == (340, 255)
    assert pp.resized_size(450, 600) == (255, 340)
    assert pp.resized_size(255, 255) == (255, 255)


def test_non_rgb_input_is_converted_first():
    rgba = _image(320, 240, seed=3).convert("RGBA")
    np.testing.assert_array_equal(pp.preprocess_image(rgba),
                                  pp.preprocess_image(rgba.convert("RGB")))


def test_batch_helper_stacks_and_reads_paths(tmp_path):
    paths = []
    for i, size in enumerate(SIZES[:3]):
        path = tmp_path / f"img{i}.png"
        _image(*size, seed=i).save(path)
        paths.append(path)
    batch = pp.preprocess_batch(paths)
    assert batch.shape == (3, 3, 224, 224) and batch.dtype == np.float32
    np.testing.assert_array_equal(batch[1], pp.preprocess_image(Image.open(paths[1])))


def test_spec_documents_the_transform():
    spec = pp.PREPROCESSING
    assert spec["resize_shorter_side"] == 255 and spec["center_crop"] == 224
    assert spec["interpolation"] == "PIL.Image.BILINEAR"
    assert spec["mean"] == [0.485, 0.456, 0.406] and spec["std"] == [0.229, 0.224, 0.225]


def test_module_is_standalone_numpy_and_pil_only():
    """It is copied verbatim into the release, so it cannot import dermssl or torch."""
    code = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('p', {str(pp.__file__)!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "bad = sorted(k for k in sys.modules if k.split('.')[0] in ('torch', 'dermssl')); "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
