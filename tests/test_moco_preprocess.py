import numpy as np
import pytest
from PIL import Image, ImageOps

from dermssl.release.moco_preprocess import preprocess_image


MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[None, None, :]
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[None, None, :]


def _reference(image):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    resized = rgb.resize((224, 224), Image.Resampling.BICUBIC)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    return ((pixels - MEAN) / STD).transpose(2, 0, 1)[None]


def test_preprocess_returns_exact_rgb_imagenet_nchw_without_center_crop():
    image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (91, 173, 3), dtype=np.uint8))

    actual = preprocess_image(image)

    assert actual.shape == (1, 3, 224, 224)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, _reference(image))


def test_preprocess_handles_grayscale_and_rgba():
    grayscale = Image.fromarray(np.arange(65 * 79, dtype=np.uint8).reshape(65, 79), mode="L")
    rgba = Image.fromarray(np.random.default_rng(2).integers(0, 256, (43, 87, 4), dtype=np.uint8), mode="RGBA")

    np.testing.assert_array_equal(preprocess_image(grayscale), _reference(grayscale))
    np.testing.assert_array_equal(preprocess_image(rgba), _reference(rgba))


def test_preprocess_applies_exif_orientation_before_resize():
    image = Image.new("RGB", (2, 3))
    image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255),
                   (255, 255, 0), (255, 0, 255), (0, 255, 255)])
    exif = image.getexif()
    exif[274] = 6
    image.info["exif"] = exif.tobytes()

    np.testing.assert_array_equal(preprocess_image(image), _reference(image))


def test_preprocess_matches_v5_torchvision_reference_when_available():
    try:
        from torchvision import transforms as T
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"torchvision reference unavailable: {exc}")
    image = Image.fromarray(np.random.default_rng(11).integers(0, 256, (67, 131, 3), dtype=np.uint8))
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    expected = transform(image).unsqueeze(0).numpy()
    np.testing.assert_allclose(preprocess_image(image), expected, rtol=0, atol=1e-6)
