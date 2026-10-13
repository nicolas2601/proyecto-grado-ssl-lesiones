"""Serial SLED adapter for a separately obtained, pinned private upstream checkout.

No third-party source is vendored. Loading upstream executes Python/Java and is
not part of component tests. Use a dedicated child process/environment only.
"""
from __future__ import annotations
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import tempfile
import numpy as np
from dermssl.w3_components import DegenerateCase, fuse_sled_scales, sled_location_selection

UPSTREAM_REVISION = '3213eacb65577c0d52f424369158fb56c609238e'


@contextmanager
def private_upstream(root: Path, seed: int = 42):
    """Load reviewed upstream in a fresh process; isolate cwd/temp/JVM classpath.

    Refuse dirty checkout, symlinks and existing generic module names. This is a
    provenance check, not a sandbox. No package installation or downloads occur.
    """
    root = root.resolve(strict=True)
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if revision != UPSTREAM_REVISION or dirty:
        raise ValueError('Clean pinned SLED checkout required')
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('SLED checkout must not contain symlinks')
    if any(n in sys.modules for n in ('algo', 'seg')):
        raise RuntimeError('SLED requires a fresh child process')
    import jpype
    if jpype.isJVMStarted():
        raise RuntimeError('Fresh process without an existing JVM required')
    jar = root / 'algo' / 'Merging.jar'
    if not jar.is_file():
        raise ValueError('Pinned upstream Merging.jar required')
    old_cwd = Path.cwd()
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    try:
        with tempfile.TemporaryDirectory(prefix='w3-sled-') as temp:
            os.chdir(temp)
            jpype.startJVM(jpype.getDefaultJVMPath(), classpath=[str(jar)])
            backend = import_module('seg.segmentation')
            classifier = import_module('seg.classifier')
            original_iforest = classifier.IForest
            def seeded_iforest(*args, **kwargs):
                kwargs['random_state'] = seed
                return original_iforest(*args, **kwargs)
            classifier.IForest = seeded_iforest
            yield backend
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(root))
        sys.dont_write_bytecode = old_bytecode
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
        # Do not reuse this process: JPype cannot safely restart the JVM.


def sled_parameters() -> SimpleNamespace:
    return SimpleNamespace(slic_compactness=10, adj_type='img_knn', self_tuning_k=30,
                           knn_k=50, centroid_thresh=.3, sigma=1., multi_scale=True,
                           outlier_detection='IFOREST', contamination=.1,
                           # GHT upstream defaults; verified separately before pilot.
                           ght_nu=1e30, ght_tau=.1, ght_kappa=1e-30, ght_omega=.9)


def run_sled_rgb(rgb: np.ndarray, valid: np.ndarray, backend, args: SimpleNamespace,
                 *, scales: tuple[int, ...] = tuple(range(200, 700, 50)),
                 selection_variant: str = 'gaussian_corrected') -> tuple[np.ndarray, np.ndarray, dict]:
    """Call upstream single-scale SE/IForest; replace fusion and final selection.

    Must be called within private_upstream in an isolated one-case process.
    No GT argument, no evaluation, no image upload. Cwd isolation alone does not
    bound upstream resource consumption: caller must enforce process timeout/RAM.
    The original label-zero centroid behavior is retained, not silently fixed.
    """
    rgb = np.asarray(rgb)
    valid = np.asarray(valid)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or valid.shape != rgb.shape[:2]:
        raise ValueError('uint8 RGB and matching validity mask required')
    if not np.isin(valid, [0, 1]).all() or not valid.any():
        raise DegenerateCase('Nonempty binary validity mask required')
    if not scales or len(set(scales)) != len(scales) or any(type(s) is not int or s <= 0 for s in scales):
        raise ValueError('Unique positive superpixel scales required')
    score_maps, variances = [], []
    for scale in scales:
        score, _, variance, _ = backend.SLED_single_scale(rgb, valid, 'case', scale, args)
        score = np.asarray(score)
        if score.shape != valid.shape:
            raise DegenerateCase('Upstream score geometry differs from original input')
        score_maps.append(score)
        variances.append(float(variance))
    fused = fuse_sled_scales(score_maps, variances)
    threshold = float(backend.ght_thresh_img(fused[valid.astype(bool)].ravel(), args))
    if not np.isfinite(threshold):
        raise DegenerateCase('Nonfinite GHT threshold')
    result = sled_location_selection(fused >= threshold, valid,
                                     variant=selection_variant, constrain_filled_holes=True)
    return result, fused, {'scales': list(scales), 'selection_variant': selection_variant,
                           'ght_threshold': threshold, 'max_workers': 1,
                           'hole_fill_clipped_to_valid': True, 'centroid_zero_behavior': 'upstream_retained'}
