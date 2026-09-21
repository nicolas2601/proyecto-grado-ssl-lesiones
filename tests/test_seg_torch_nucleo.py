"""Torch-dependent tests for GPU augmentation, distributed helpers and checkpoints."""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch

import seg_paths  # noqa: F401,E402
import ckpt_io  # noqa: E402
import dist_utils  # noqa: E402
import gpu_augment as ga  # noqa: E402


def _generator(seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _batch(batch=4, size=16):
    return torch.rand(batch, 3, size, size, generator=_generator(1))


def test_to_float_batch_moves_channels_first_and_scales():
    raw = torch.full((2, 8, 8, 3), 255, dtype=torch.uint8)
    out = ga.to_float_batch(raw)
    assert out.shape == (2, 3, 8, 8) and out.dtype == torch.float32
    assert float(out.max()) == 1.0


def test_normalize_batch_uses_imagenet_statistics():
    out = ga.normalize_batch(torch.full((1, 3, 2, 2), 0.485))
    assert abs(float(out[0, 0, 0, 0])) < 1e-5


def test_pure_flip_keeps_image_and_mask_aligned():
    images = _batch()
    masks = images[:, :1].clone()
    out_images, out_masks = ga.random_resized_crop(
        images, _generator(2), scale=(1.0, 1.0), ratio=(1.0, 1.0), masks=masks,
        vertical_flip=True,
    )
    assert torch.allclose(out_images[:, 0:1], out_masks, atol=1e-5)
    assert out_images.shape == images.shape


def test_crop_is_deterministic_for_a_given_generator_seed():
    images = _batch()
    first = ga.random_resized_crop(images, _generator(5), scale=(0.3, 1.0))[0]
    second = ga.random_resized_crop(images, _generator(5), scale=(0.3, 1.0))[0]
    assert torch.equal(first, second)


def test_crop_masks_keep_only_original_label_values():
    images = _batch()
    masks = torch.randint(0, 2, (4, 1, 16, 16), generator=_generator(3)).float() * 254 + 1
    masks[masks > 200] = 255.0
    _, out_masks = ga.random_resized_crop(images, _generator(4), scale=(0.3, 1.0), masks=masks)
    assert set(torch.unique(out_masks).tolist()) <= set(torch.unique(masks).tolist())


def test_color_jitter_with_zero_strength_is_identity():
    images = _batch()
    out = ga.color_jitter(images, _generator(), 0.0, 0.0, 0.0, 0.0, probability=1.0)
    assert torch.allclose(out, images, atol=1e-5)


def test_color_jitter_stays_in_unit_range():
    out = ga.color_jitter(_batch(), _generator(), 0.4, 0.4, 0.4, 0.1, probability=1.0)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_gaussian_blur_smooths_and_probability_zero_is_identity():
    images = _batch(size=32)
    blurred = ga.gaussian_blur(images, _generator(), probability=1.0)
    untouched = ga.gaussian_blur(images, _generator(), probability=0.0)
    assert float(blurred.var()) < float(images.var())
    assert torch.equal(untouched, images)


def test_grayscale_probability_one_equalizes_channels():
    out = ga.random_grayscale(_batch(), _generator(), probability=1.0)
    assert torch.allclose(out[:, 0], out[:, 1], atol=1e-6)


def test_moco_and_student_augment_preserve_shapes():
    images = _batch(size=32)
    assert ga.moco_augment(images, _generator()).shape == images.shape
    masks = torch.zeros(4, 1, 32, 32)
    out_images, out_masks = ga.student_augment(images, masks, _generator())
    assert out_images.shape == images.shape and out_masks.shape == masks.shape


def test_single_process_context_is_rank_zero_on_cpu():
    context = dist_utils.init_distributed(force_cpu=True)
    assert context.rank == 0 and context.world_size == 1 and context.is_main
    tensor = torch.tensor([2.0])
    assert torch.equal(dist_utils.all_reduce_sum(tensor, context), tensor)
    assert dist_utils.all_gather_cat(tensor, context).shape == (1,)
    dist_utils.barrier(context)
    dist_utils.cleanup(context)


def test_checkpoint_roundtrip_is_atomic(tmp_path):
    path = tmp_path / "ckpt" / "last.pt"
    ckpt_io.save_checkpoint(path, {"epoch": 3, "weights": torch.ones(2), "selector": {"a": 1.5}})
    loaded = ckpt_io.load_checkpoint(path)
    assert loaded["epoch"] == 3 and loaded["selector"]["a"] == 1.5
    assert not list(path.parent.glob("*.tmp"))


def test_vignette_darkens_the_border_and_keeps_the_centre():
    images = torch.ones(4, 3, 32, 32)
    out = ga.random_vignette(images, _generator(3), probability=1.0)
    assert float(out[:, :, 16, 16].mean()) > 0.99
    assert float(out[:, :, 0, 0].mean()) < 0.8
    assert float(out[:, :, 0, 0].max()) < float(out[:, :, 16, 16].min())


def test_vignette_probability_zero_is_identity():
    images = _batch(size=32)
    assert torch.equal(ga.random_vignette(images, _generator(), probability=0.0), images)


def test_vignette_stays_in_unit_range_and_is_reproducible():
    images = _batch(size=32)
    first = ga.random_vignette(images, _generator(7), probability=1.0)
    second = ga.random_vignette(images, _generator(7), probability=1.0)
    assert torch.equal(first, second)
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0


def test_vignette_never_brightens_a_pixel():
    images = _batch(size=32)
    out = ga.random_vignette(images, _generator(5), probability=1.0)
    assert bool((out <= images + 1e-6).all())


def test_moco_and_student_views_include_the_vignette():
    images = torch.ones(64, 3, 32, 32)
    moco = ga.moco_augment(images, _generator(11))
    assert float(moco[:, :, 0, 0].mean()) < float(moco[:, :, 16, 16].mean())
    masks = torch.full((64, 1, 32, 32), 255.0)
    student, student_masks = ga.student_augment(images, masks, _generator(13))
    assert float(student[:, :, 0, 0].mean()) < float(student[:, :, 16, 16].mean())
    assert set(torch.unique(student_masks).tolist()) == {255.0}
