"""Static guarantees for the segmentation pipeline (no torch needed).

* every module compiles;
* CLIs validate arguments before importing torch;
* no training-side code can read manual masks or labels, and no code selects
  a checkpoint or threshold with ground truth: only the evaluation module
  opens reference masks.
"""

import ast
import py_compile
import re

import pytest

import seg_paths

ALL_MODULES = sorted(p for p in seg_paths.PLAN.rglob("*.py"))
EVALUATION_MODULE = seg_paths.COMUN / "evaluacion" / "evaluate_segmentation.py"
# synthetic.py only WRITES fake reference masks for the smoke test; it never reads any.
TRAINING_SIDE = [
    p for p in ALL_MODULES if p not in (EVALUATION_MODULE, seg_paths.COMUN / "nucleo" / "synthetic.py")
]
CLI_SCRIPTS = [
    seg_paths.COMUN / "datos" / "prepare_cache.py",
    seg_paths.COMUN / "estudiante" / "train_student.py",
    seg_paths.COMUN / "evaluacion" / "evaluate_segmentation.py",
    seg_paths.CAMINO_B / "preentrenamiento-contrastivo" / "train_moco.py",
    seg_paths.CAMINO_B / "pseudo-mascaras" / "train_ccam.py",
    seg_paths.CAMINO_B / "pseudo-mascaras" / "make_pseudo_masks.py",
    seg_paths.CAMINO_B / "entrenamiento" / "run_pipeline.py",
]
FORBIDDEN_IN_TRAINING = (
    "train_unet",
    "ISIC2018SegmentationDataset",
    "_segmentation.png",
    "groundtruth",
    "ground_truth",
    "diagnosis",
    "isic2018",
)


def _module_level_imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_there_are_modules_to_check():
    assert len(ALL_MODULES) >= 20


@pytest.mark.parametrize("path", ALL_MODULES, ids=lambda p: p.name)
def test_every_module_compiles(path):
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("path", CLI_SCRIPTS, ids=lambda p: p.name)
def test_cli_scripts_do_not_import_torch_at_module_level(path):
    assert not any(name.split(".")[0] == "torch" for name in _module_level_imports(path))


@pytest.mark.parametrize("path", TRAINING_SIDE, ids=lambda p: p.name)
def test_training_side_code_never_names_masks_or_labels_of_the_test_sets(path):
    source = path.read_text(encoding="utf-8").lower()
    for token in FORBIDDEN_IN_TRAINING:
        if path.name == "run_pipeline.py" and token == "isic2018":
            continue  # docstring example paths only; see the command test
        assert token.lower() not in source, f"{path.name} mentions {token!r}"


def test_only_cache_builder_and_evaluation_open_image_files():
    openers = {
        p.name
        for p in ALL_MODULES
        if "Image.open(" in p.read_text(encoding="utf-8") and p.name != "synthetic.py"
    }
    assert openers == {"cache_io.py", "evaluate_segmentation.py", "fst_pseudo_manifest.py"}


def test_no_training_module_imports_the_evaluation_module():
    for path in TRAINING_SIDE:
        assert "evaluate_segmentation" not in " ".join(_module_level_imports(path)), path.name


def test_student_reuses_only_the_unet_architecture_from_torch_unet():
    engine = seg_paths.COMUN / "estudiante" / "student_engine.py"
    tree = ast.parse(engine.read_text(encoding="utf-8"))
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "dermssl.torch_unet"
        for alias in node.names
    ]
    assert imported == ["UNet"]


def test_no_hard_coded_absolute_paths_in_pipeline_code():
    pattern = re.compile(r"[\"'](?:[A-Za-z]:[\/]|/(?:home|scratch|datos|mnt|kaggle)/)")
    for path in ALL_MODULES:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("#", '"""', "python", "torchrun", "--")):
                continue
            assert not pattern.search(line), f"{path.name}: {stripped}"


def test_checkpoint_selection_never_receives_reference_masks():
    engine = (seg_paths.COMUN / "estudiante" / "student_engine.py").read_text(encoding="utf-8")
    assert "selector.update(epoch, val_loss)" in engine
    assert "evaluate" not in engine.lower().replace("evaluation", "")


def test_final_threshold_is_fixed_at_one_half():
    source = EVALUATION_MODULE.read_text(encoding="utf-8")
    assert "DEFAULT_THRESHOLD = 0.5" in source
    assert "--threshold" not in source


def test_segmentation_training_does_not_use_dataloader_workers():
    for path in CLI_SCRIPTS:
        if path.name in {"train_moco.py", "train_ccam.py", "train_student.py", "run_pipeline.py"}:
            assert "DataLoader" not in path.read_text(encoding="utf-8")
            assert "num_workers" not in path.read_text(encoding="utf-8")
