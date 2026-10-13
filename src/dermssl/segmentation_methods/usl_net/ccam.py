"""Reuse the reviewed v8 CCAM objective; keep historical training untouched."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def existing_ccam_loss(features, foreground, alpha=.25):
    root = Path(__file__).resolve().parents[4]
    path = root/'plan-segmentacion/camino-b-cnn/pseudo-mascaras/ccam_engine.py'
    spec = spec_from_file_location('v8_reviewed_ccam_objective',path)
    module = module_from_spec(spec); spec.loader.exec_module(module)
    return module.ccam_loss(features,foreground,alpha)
