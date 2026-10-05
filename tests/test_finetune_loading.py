"""The FT data loader must not need /dev/shm: workers hand back numpy, never torch tensors."""

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402
from torchvision import transforms  # noqa: E402

from dermssl.finetune import loading, torch_run  # noqa: E402

BILINEAR = transforms.InterpolationMode.BILINEAR


@pytest.fixture
def dataset_parts(tmp_path):
    ids = [f"img{k:02d}" for k in range(12)]
    rng = np.random.default_rng(0)
    paths = {}
    for i in ids:
        paths[i] = tmp_path / f"{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (48, 40, 3), dtype=np.uint8)).save(paths[i])
    return ids, paths, {i: k % 2 for k, i in enumerate(ids)}


def _reference_batches(parts, train, workers, seed):
    ids, paths, labels = parts
    tf_train, tf_eval = torch_run.build_transforms(BILINEAR)
    ds = torch_run.FTDataset(ids, paths, labels, tf_train if train else tf_eval)
    torch.manual_seed(seed)
    return list(DataLoader(ds, batch_size=4, shuffle=train, num_workers=workers))


def _new_batches(parts, train, workers, seed):
    ids, paths, labels = parts
    tf_train, tf_eval = loading.build_worker_transforms(BILINEAR)
    ds = torch_run.FTDataset(ids, paths, labels, tf_train if train else tf_eval)
    torch.manual_seed(seed)
    dl = DataLoader(ds, batch_size=4, shuffle=train, num_workers=workers,
                    collate_fn=loading.collate_uint8)
    return [(loading.to_model_input(x, "cpu"), loading.to_labels(y, "cpu")) for x, y in dl]


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("workers", [0, 2])
def test_batches_are_bit_identical_to_the_tensor_pipeline(dataset_parts, train, workers):
    old = _reference_batches(dataset_parts, train, workers, seed=42)
    new = _new_batches(dataset_parts, train, workers, seed=42)
    assert len(old) == len(new) == 3
    for (xo, yo), (xn, yn) in zip(old, new):
        assert xn.dtype == torch.float32 and xn.shape == xo.shape
        assert torch.equal(xo, xn)
        assert torch.equal(yo, yn)


def test_workers_return_numpy_not_shared_tensors(dataset_parts):
    ids, paths, labels = dataset_parts
    tf_train, _ = loading.build_worker_transforms(BILINEAR)
    ds = torch_run.FTDataset(ids, paths, labels, tf_train)
    xb, yb = loading.collate_uint8([ds[0], ds[1]])
    assert isinstance(xb, np.ndarray) and xb.dtype == np.uint8 and xb.shape == (2, 224, 224, 3)
    assert isinstance(yb, np.ndarray) and yb.dtype == np.int64


def test_collate_never_builds_a_torch_tensor(dataset_parts, monkeypatch):
    ids, paths, labels = dataset_parts
    _, tf_eval = loading.build_worker_transforms(BILINEAR)
    ds = torch_run.FTDataset(ids, paths, labels, tf_eval)
    samples = [ds[0], ds[1]]

    def boom(*a, **k):
        raise AssertionError("collate must not touch torch (its tensors go through /dev/shm)")

    monkeypatch.setattr(torch, "stack", boom)
    monkeypatch.setattr(torch, "tensor", boom)
    monkeypatch.setattr(torch, "as_tensor", boom)
    monkeypatch.setattr(torch, "from_numpy", boom)
    loading.collate_uint8(samples)


def _sha(batches) -> str:
    import hashlib
    h = hashlib.sha256()
    for x, y in batches:
        h.update(x.numpy().tobytes() + y.numpy().tobytes())
    return h.hexdigest()


def test_eval_batches_have_the_same_sha256_for_the_default_collate_without_workers(dataset_parts):
    """Default collate, num_workers=0 (no shm) vs numpy collate, num_workers=2."""
    ref = _sha(_reference_batches(dataset_parts, train=False, workers=0, seed=42))
    assert _sha(_new_batches(dataset_parts, train=False, workers=2, seed=42)) == ref


def test_train_augmentation_draws_depend_on_the_worker_count_in_the_original_pipeline_too(
        dataset_parts):
    """Evidence for the limit of 'bit a bit': PyTorch seeds each worker, so random augmentations
    differ between num_workers=0 and >0 for BOTH pipelines. The new one matches the original at
    equal worker count (test above), which is the claim that can be made."""
    w0 = _sha(_reference_batches(dataset_parts, train=True, workers=0, seed=42))
    w2 = _sha(_reference_batches(dataset_parts, train=True, workers=2, seed=42))
    assert w0 != w2
