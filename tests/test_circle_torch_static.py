"""``finetune/circle_torch.py`` cannot run here (no torch, no GPU) but CAN be
checked: it must compile, the frozen-backbone/probe-only-trainable contract
must be visible at the source level, and it must not leak torch into the
pure package. Modeled on ``tests/test_torch_unet_static.py``, itself modeled
on ``tests/test_pretrain_torch_layer.py``.

This is Stage 2's GPU-side machinery, built ahead of the real Stage 1 Kaggle
run: these tests check the joint-regularization CONTRACT (backbone frozen,
probe-only optimizer, MSE term wired in), not a real training result - no
real Stage 1 data exists yet to train against.
"""

import ast
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CIRCLE_TORCH = ROOT / "src" / "dermssl" / "finetune" / "circle_torch.py"


def _tree():
    return ast.parse(CIRCLE_TORCH.read_text())


def _function(name: str) -> ast.FunctionDef:
    return next(
        n for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _class(name: str) -> ast.ClassDef:
    return next(n for n in ast.walk(_tree()) if isinstance(n, ast.ClassDef) and n.name == name)


def test_circle_torch_module_exists():
    assert CIRCLE_TORCH.is_file()


def test_circle_torch_compiles():
    """The cheapest possible proof the cluster/Kaggle session will not die on
    a SyntaxError."""
    py_compile.compile(str(CIRCLE_TORCH), doraise=True)


def test_importing_dermssl_never_imports_torch():
    """The v5/v6 selling point stands for Stage 2 too: the unit suite needs
    no GPU stack. ``circle_torch.py`` sits beside ``torch_run.py`` in
    ``finetune/`` but importing the top package (or the finetune subpackage)
    must never pull in torch."""
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch."):
            del sys.modules[name]
    import dermssl  # noqa: F401
    import dermssl.finetune  # noqa: F401

    assert "torch" not in sys.modules


def test_circle_torch_not_wired_into_module_order():
    """GPU-only (design.md File Changes table); must never be inlined by the
    notebook-build exec test."""
    import dermssl

    assert "circle_torch" not in dermssl.MODULE_ORDER


def test_freeze_backbone_sets_requires_grad_false_and_evals():
    """spec.md req 3: 'The backbone MUST remain frozen (no weight updates)
    throughout Stage 2 training.' Source-level check since we cannot
    construct a real nn.Module here: the function must iterate every backbone
    parameter and set requires_grad = False, and put the backbone in eval
    mode (matches D3-style two-tier contracts elsewhere in this repo)."""
    fn = _function("freeze_backbone")
    source = ast.get_source_segment(CIRCLE_TORCH.read_text(), fn)
    assert "requires_grad = False" in source
    assert ".parameters()" in source
    assert ".eval()" in source


def test_probe_head_class_exposes_forward():
    cls = _class("ProbeHead")
    methods = [n.name for n in ast.walk(cls) if isinstance(n, ast.FunctionDef)]
    assert "forward" in methods
    assert "__init__" in methods


def test_joint_regularization_loss_combines_task_loss_and_mse():
    """spec.md req 3: MSE regularization between the probe's prediction on
    the original image and its FST-shifted counterpart, added to the task
    loss - not a replacement for it."""
    fn = _function("joint_regularization_loss")
    source = ast.get_source_segment(CIRCLE_TORCH.read_text(), fn)
    assert "mse_loss" in source
    args = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
    assert "criterion" in args
    assert "reg_weight" in args


def test_build_probe_optimizer_only_passes_probe_parameters():
    """The optimizer must never see backbone parameters - the invariant that
    keeps 'only probe-head parameters receive updates' true regardless of
    what freeze_backbone already enforces."""
    fn = _function("build_probe_optimizer")
    source = ast.get_source_segment(CIRCLE_TORCH.read_text(), fn)
    assert "probe.parameters()" in source
    assert "backbone.parameters()" not in source


def test_stage2_train_step_asserts_backbone_frozen_before_optimizing():
    """Order matters: the frozen-backbone assertion must run before any
    gradient step, mirroring scripts/pretrain.py's validate-before-import
    pattern (test_launch_script_validates_config_before_importing_torch)."""
    fn = _function("stage2_train_step")
    source = ast.get_source_segment(CIRCLE_TORCH.read_text(), fn)
    assert "requires_grad" in source
    assert source.index("requires_grad") < source.index("optimizer.step()")


def test_stage2_train_step_computes_backbone_features_under_no_grad():
    """Backbone features for both the original and FST-shifted image must be
    computed under torch.no_grad(): the backbone contributes no graph, so no
    backbone weight can receive a gradient even indirectly."""
    fn = _function("stage2_train_step")
    source = ast.get_source_segment(CIRCLE_TORCH.read_text(), fn)
    assert "no_grad" in source


def test_stage2_train_step_only_backward_through_the_probe():
    fn = _function("stage2_train_step")
    source = ast.get_source_segment(CIRCLE_TORCH.read_text(), fn)
    assert "backward()" in source
    assert "optimizer.step()" in source


def test_stage2_train_step_frozen_invariant_is_not_a_bare_assert():
    """A bare ``assert`` is stripped entirely under ``python -O`` /
    ``PYTHONOPTIMIZE=1``, silently defeating the exact defense-in-depth
    promise ``stage2_train_step``'s docstring makes ("fails loudly instead
    of silently updating the backbone"). The frozen-backbone check must be
    an explicit ``if`` + ``raise``, mirroring fitzpatrick.py's
    ita_degrees/estimate_fst fix for the same class of issue (PR1)."""
    fn = _function("stage2_train_step")
    assert not any(isinstance(n, ast.Assert) for n in ast.walk(fn)), (
        "stage2_train_step must not enforce the frozen-backbone invariant "
        "with a bare assert - it is stripped under python -O"
    )


def test_stage2_train_step_raises_explicit_error_when_backbone_not_frozen():
    """The invariant must still fail loudly: an explicit conditional raising
    a specific exception, checked before the optimizer is touched."""
    fn = _function("stage2_train_step")
    source = ast.get_source_segment(CIRCLE_TORCH.read_text(), fn)
    assert "requires_grad" in source
    assert "raise BackboneNotFrozenError" in source
    assert source.index("requires_grad") < source.index("optimizer.step()")
    assert source.index("raise BackboneNotFrozenError") < source.index("optimizer.step()")


def test_backbone_not_frozen_error_is_a_runtime_error_subclass():
    """Defined at module scope so callers can catch it specifically instead
    of a bare ``RuntimeError``, matching the ``LabelLeakError(RuntimeError)``
    convention in ``pretrain/config.py``."""
    cls = _class("BackboneNotFrozenError")
    bases = [b.id for b in cls.bases if isinstance(b, ast.Name)]
    assert bases == ["RuntimeError"]
