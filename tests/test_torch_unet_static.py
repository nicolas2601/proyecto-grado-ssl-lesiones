"""``torch_unet.py`` and the Kaggle FST-manifest script cannot run here (no
torch, no GPU) but CAN be checked: they must compile, and the GPU code must
not leak into the pure package - D2/D4 in design.md: ``dermssl``'s tested
core stays numpy-only, the GPU/CPU seam is ``fst_manifest.csv``, never an
in-process import. Modeled on ``tests/test_pretrain_torch_layer.py``."""

import ast
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TORCH_UNET = ROOT / "src" / "dermssl" / "torch_unet.py"
KAGGLE_SCRIPT = ROOT / "scripts" / "kaggle_fst_manifest.py"


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


def test_train_unet_accepts_and_seeds_a_reproducibility_seed():
    """The Kaggle U-Net run feeds ``gates.fairness_gate`` (D4): weight init,
    batch shuffling, and cuDNN kernel selection are all nondeterministic
    without an explicit seed, so re-running the SAME manifest input (e.g.
    recovering from a Kaggle timeout) could silently change the fairness
    gate's PASS/FAIL outcome. ``train_unet`` must accept a documented
    ``seed`` default and actually call ``torch.manual_seed`` with it."""
    tree = ast.parse(TORCH_UNET.read_text())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "train_unet"
    )
    all_args = fn.args.args + fn.args.kwonlyargs
    arg_names = [a.arg for a in all_args]
    assert "seed" in arg_names, "train_unet must accept a seed parameter"

    # The seed must have a documented default (positional defaults align to
    # the tail of fn.args.args; kwonly defaults align 1:1 with kwonlyargs).
    seed_index = arg_names.index("seed")
    if seed_index < len(fn.args.args):
        pos_defaults_start = len(fn.args.args) - len(fn.args.defaults)
        assert seed_index >= pos_defaults_start, "seed must have a default value"
    else:
        kwonly_index = seed_index - len(fn.args.args)
        assert fn.args.kw_defaults[kwonly_index] is not None, (
            "seed must have a default value"
        )

    # The seed must actually be used: torch.manual_seed(...) somewhere in
    # the function body, not just accepted and ignored.
    calls_manual_seed = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "manual_seed"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        for node in ast.walk(fn)
    )
    assert calls_manual_seed, "train_unet must call torch.manual_seed(seed)"


def test_kaggle_script_dataloader_uses_a_seeded_generator():
    """``DataLoader(..., shuffle=True)`` with no seeded generator makes batch
    order nondeterministic across reruns of the same manifest input - the
    same reproducibility gap as an unseeded ``train_unet`` (D4)."""
    source = KAGGLE_SCRIPT.read_text()
    assert "--seed" in source, "the script must expose a --seed CLI flag"
    assert "torch.Generator(" in source
    assert "manual_seed" in source
    assert "generator=" in source, "DataLoader must receive the seeded generator"


def test_kaggle_script_mask_source_reuses_fitzpatrick_degeneracy_check():
    """``mask_source`` labeling must call fitzpatrick's shared degeneracy
    check directly rather than re-deriving the area-bounds comparison
    inline - otherwise a change to fitzpatrick.py's bounds or comparison
    operators would silently drift from the label the script writes to
    fst_manifest.csv's mask_source column."""
    source = KAGGLE_SCRIPT.read_text()
    assert "fitzpatrick.is_degenerate_mask(" in source, (
        "the script must call the shared degeneracy check, not reimplement it"
    )
    assert "MIN_MASK_AREA_FRACTION" not in source
    assert "MAX_MASK_AREA_FRACTION" not in source
