"""``torch_unet.py`` and the Kaggle FST-manifest script cannot run here (no
torch, no GPU) but CAN be checked: they must compile, and the GPU code must
not leak into the pure package - D2/D4 in design.md: ``dermssl``'s tested
core stays numpy-only, the GPU/CPU seam is ``fst_manifest.csv``, never an
in-process import. Modeled on ``tests/test_pretrain_torch_layer.py``."""

import ast
import importlib.util
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TORCH_UNET = ROOT / "src" / "dermssl" / "torch_unet.py"
KAGGLE_SCRIPT = ROOT / "scripts" / "kaggle_fst_manifest.py"


def _load_kaggle_script():
    spec = importlib.util.spec_from_file_location("kaggle_fst_manifest", KAGGLE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_torch_unet_module_exists():
    assert TORCH_UNET.is_file()


def test_torch_unet_compiles():
    """The cheapest possible proof the cluster/Kaggle session will not die on
    a SyntaxError."""
    py_compile.compile(str(TORCH_UNET), doraise=True)


def test_kaggle_fst_manifest_script_exists():
    assert KAGGLE_SCRIPT.is_file()


def test_kaggle_fst_manifest_script_compiles():
    py_compile.compile(str(KAGGLE_SCRIPT), doraise=True)


def test_importing_dermssl_never_imports_torch():
    """The v5/v6 selling point stands for Phase 3 too: the unit suite needs
    no GPU stack. ``torch_unet.py`` sits beside ``gates.py``/``metrics.py``
    but is NOT in ``MODULE_ORDER`` and NOT imported by ``__init__.py``, so
    importing the top package must never pull in torch."""
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            del sys.modules[name]
    import dermssl  # noqa: F401

    assert "torch" not in sys.modules


def test_torch_unet_not_wired_into_module_order():
    """``torch_unet.py`` is GPU-only (design.md File Changes table); it must
    never be inlined/executed by the notebook-build exec test."""
    import dermssl

    assert "torch_unet" not in dermssl.MODULE_ORDER


def test_pure_modules_do_not_mention_torch_in_imports():
    """Mirrors test_pretrain_torch_layer.py's guard for the pretrain package,
    applied to dermssl's top-level modules (excluding the torch_ prefixed
    ones and the finetune/pretrain subpackages, which have their own guards)."""
    pure = [
        p
        for p in (ROOT / "src" / "dermssl").glob("*.py")
        if not p.name.startswith("torch_")
    ]
    assert len(pure) >= 8  # __init__ + fitzpatrick + gates + metrics + ...
    for path in pure:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("torch") for a in node.names), path.name
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("torch"), path.name


def test_kaggle_script_validates_args_before_importing_torch():
    """Mirrors scripts/pretrain.py: argparse first, ``import torch`` only
    inside ``run()``, so a bad flag fails in milliseconds on a machine with
    no GPU stack instead of after a Kaggle session queue wait."""
    source = KAGGLE_SCRIPT.read_text()
    before_run = source.split("def run(")[0]
    assert "import torch" not in before_run
    assert "from dermssl import torch_unet" not in before_run


def test_unet_class_exposes_forward():
    tree = ast.parse(TORCH_UNET.read_text())
    cls = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "UNet"
    )
    methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
    assert "forward" in methods


def test_infer_mask_has_the_estimate_fst_mask_contract():
    """``infer_mask`` must accept a model + image - the exact ``mask``
    contract ``fitzpatrick.estimate_fst`` expects (D3): bool (H, W), True =
    lesion."""
    tree = ast.parse(TORCH_UNET.read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "infer_mask"
    )
    args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    assert "model" in args
    assert "img" in args


def test_train_unet_and_checkpoint_helpers_exist():
    tree = ast.parse(TORCH_UNET.read_text())
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"train_unet", "save_checkpoint", "load_checkpoint"} <= names


def test_train_unet_requires_held_out_validation_and_selects_best_dice_checkpoint():
    source = TORCH_UNET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_unet"
    )
    args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    fn_source = ast.get_source_segment(source, fn)

    assert "val_loader" in args
    assert "evaluate_segmentation" in fn_source
    assert "best_val_dice" in fn_source
    assert "load_state_dict" in fn_source


def test_torch_unet_exposes_validation_metrics_and_mask_confidence_contracts():
    tree = ast.parse(TORCH_UNET.read_text(encoding="utf-8"))
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}

    assert {"evaluate_segmentation", "infer_mask_with_confidence"} <= functions
    assert {"UNetTrainingResult", "MaskPrediction"} <= classes


def test_isic2018_dataset_class_exists():
    tree = ast.parse(TORCH_UNET.read_text())
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "ISIC2018SegmentationDataset" in classes


def test_kaggle_script_writes_the_manifest_schema_columns():
    """D4: the GPU/CPU seam is ``fst_manifest.csv`` with exactly these
    columns (design.md, "GPU/CPU seam" + tasks.md 3.2)."""
    source = KAGGLE_SCRIPT.read_text()
    for column in (
        "image_id",
        "ita",
        "fst_bucket",
        "skin_fraction",
        "mask_source",
        "mask_confidence",
        "fallback_used",
        "confidence",
        "artifact_detected",
        "status",
    ):
        assert f'"{column}"' in source


def test_kaggle_script_wires_infer_mask_into_estimate_fst():
    """The script must feed the U-Net's mask into ``estimate_fst`` (D3:
    U-Net is primary, Otsu inside ``estimate_fst`` is the fallback) rather
    than reimplementing FST math."""
    source = KAGGLE_SCRIPT.read_text()
    assert "infer_mask" in source
    assert "estimate_fst" in source
    assert "fitzpatrick" in source


def test_kaggle_script_builds_a_deterministic_held_out_validation_loader():
    source = KAGGLE_SCRIPT.read_text(encoding="utf-8")

    assert "deterministic_split_indices" in source
    assert "train_indices" in source
    assert "val_indices" in source
    assert "val_loader" in source
    assert "seed=args.seed" in source


def test_deterministic_split_indices_is_reproducible_disjoint_and_nonempty():
    script = _load_kaggle_script()

    train_a, val_a = script.deterministic_split_indices(10, val_fraction=0.2, seed=7)
    train_b, val_b = script.deterministic_split_indices(10, val_fraction=0.2, seed=7)

    assert (train_a, val_a) == (train_b, val_b)
    assert len(train_a) == 8 and len(val_a) == 2
    assert set(train_a).isdisjoint(val_a)
    assert sorted(train_a + val_a) == list(range(10))


def test_deterministic_split_indices_changes_with_seed():
    script = _load_kaggle_script()

    _, val_a = script.deterministic_split_indices(20, val_fraction=0.2, seed=7)
    _, val_b = script.deterministic_split_indices(20, val_fraction=0.2, seed=8)

    assert val_a != val_b


def test_kaggle_script_propagates_mask_confidence_and_explicit_unavailable():
    source = KAGGLE_SCRIPT.read_text(encoding="utf-8")

    assert "infer_mask_with_confidence" in source
    assert "mask_confidence=prediction.confidence" in source
    assert '"unavailable"' in source
