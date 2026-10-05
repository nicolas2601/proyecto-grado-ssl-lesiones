"""C1's geometry against torchvision, on REAL HAM10000 images.

Every other geometry test in this suite runs on synthetic arrays, which is why
the centre-crop offset survived 1500 green tests: the synthetic cases were
square, so the odd margin that HAM10000 produces never appeared.

This one needs the actual dataset and torchvision, so it skips everywhere
except the pod and any CI that has them. That is the point -- it is the test
that only passes when the thing it protects is really true, on the images the
experiment actually uses.

Run on the pod with:

    DERMSSL_HAM_IMAGE_DIR=~/work/proyectonp/eval_ham10000_images \\
        python -m pytest tests/test_c1_geometry_on_real_images.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dermssl import c1_extraction as ex
from dermssl import mask_conditions as mc

IMAGE_DIR_VAR = "DERMSSL_HAM_IMAGE_DIR"
N_IMAGES = 20


def _image_paths() -> list[Path]:
    raw = os.environ.get(IMAGE_DIR_VAR, "")
    if not raw:
        pytest.skip(f"{IMAGE_DIR_VAR} sin definir: este test necesita las imagenes reales")
    directory = Path(raw).expanduser()
    if not directory.is_dir():
        # Not a skip: the variable was set on purpose, so a wrong path is an
        # error. Skipping here would report "todo bien" for a test that never
        # compared anything.
        raise AssertionError(f"{IMAGE_DIR_VAR}={raw!r} no es un directorio")

    paths = sorted(directory.glob("*.jpg"))[:N_IMAGES]
    assert len(paths) == N_IMAGES, (
        f"esperaba {N_IMAGES} imagenes en {directory}, encontre {len(paths)}"
    )
    return paths


@pytest.fixture(scope="module")
def images() -> list[Path]:
    return _image_paths()


@pytest.fixture(scope="module")
def reference():
    transforms = pytest.importorskip("torchvision.transforms")
    return transforms.Compose(
        [
            transforms.Resize(
                ex._INTERMEDIATE, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(ex.RESOLUTION),
        ]
    )


class TestImageGeometryIsExact:
    def test_every_image_matches_torchvision_pixel_for_pixel(self, images, reference):
        from PIL import Image

        mismatched = []
        for path in images:
            pil = Image.open(path).convert("RGB")
            expected = np.asarray(reference(pil), dtype=np.uint8)
            actual = ex._baseline_geometry_image(np.asarray(pil, dtype=np.uint8))

            if not np.array_equal(actual, expected):
                differing = int((actual != expected).sum())
                mismatched.append(f"{path.name}: {differing} px distintos")

        assert not mismatched, "\n".join(mismatched)

    def test_the_images_are_rectangular_so_the_margin_is_really_odd(self, images):
        """Guards the test itself.

        If this ever runs on a square-cropped copy of HAM10000, the exactness
        test above would pass without exercising the case it exists for, and
        would keep passing while the bug came back.
        """
        from PIL import Image

        shapes = {Image.open(path).size for path in images}
        assert shapes, "no se leyo ninguna imagen"
        for width, height in shapes:
            assert width != height, f"imagen cuadrada {width}x{height}: el margen seria par"

        short_side = min(min(shape) for shape in shapes)
        long_side = max(max(shape) for shape in shapes)
        resized_short = ex._INTERMEDIATE
        resized_long = int(ex._INTERMEDIATE * long_side / short_side)
        assert (resized_short - ex.RESOLUTION) % 2 == 1, (
            "el margen del lado corto es par: este test no prueba el caso que importa"
        )
        assert resized_long > ex.RESOLUTION


class TestMaskTravelsWithTheImage:
    def _disc(self, height: int, width: int, centre, radius: int) -> np.ndarray:
        rows, columns = np.ogrid[:height, :width]
        return ((rows - centre[0]) ** 2 + (columns - centre[1]) ** 2) <= radius**2

    def test_mask_and_image_keep_the_same_crop_offsets(self, images):
        """Same geometry, or condition (c) pools the wrong patches.

        A mask cropped one row off the image would still be 224x224 and still
        look like a lesion -- the failure is silent, which is why it is pinned
        against the image's own offsets rather than against a shape.
        """
        from PIL import Image

        for path in images:
            pil = Image.open(path).convert("RGB")
            width, height = pil.size
            mask = self._disc(height, width, (height // 2, width // 3), min(height, width) // 6)

            cropped_mask = ex._baseline_geometry_mask(mask)
            cropped_image = ex._baseline_geometry_image(np.asarray(pil, dtype=np.uint8))

            assert cropped_mask.shape == cropped_image.shape[:2], path.name

            # The marker travels through the IMAGE path; if the two paths used
            # different offsets, it would land somewhere else than the mask.
            marker = np.zeros((height, width, 3), dtype=np.uint8)
            marker[mask] = 255
            marker_after_image_path = ex._baseline_geometry_image(marker)[:, :, 0] > 127

            overlap = (marker_after_image_path & cropped_mask).sum()
            union = (marker_after_image_path | cropped_mask).sum()
            assert union > 0, f"{path.name}: la lesion sintetica desaparecio entera"
            iou = overlap / union
            assert iou > 0.97, f"{path.name}: mascara e imagen desalineadas (IoU {iou:.3f})"

    def test_the_patch_grid_lands_on_the_same_cells(self, images):
        """What condition (c) actually consumes is the 16x16 patch grid."""
        from PIL import Image

        for path in images:
            pil = Image.open(path).convert("RGB")
            width, height = pil.size
            mask = self._disc(height, width, (height // 2, width // 2), min(height, width) // 5)

            marker = np.zeros((height, width, 3), dtype=np.uint8)
            marker[mask] = 255

            from_mask_path = mc.mask_to_patch_grid(ex._baseline_geometry_mask(mask), ex.GRID)
            from_image_path = mc.mask_to_patch_grid(
                ex._baseline_geometry_image(marker)[:, :, 0] > 127, ex.GRID
            )

            assert from_mask_path.shape == (ex.GRID, ex.GRID)
            assert from_mask_path.any(), f"{path.name}: ningun parche quedo marcado"
            differing = int((from_mask_path != from_image_path).sum())
            assert differing <= 4, (
                f"{path.name}: {differing} celdas del grid 16x16 no coinciden entre "
                "el camino de la mascara y el de la imagen"
            )
