# Design: Fitzpatrick Domain Adaptation (CIRCLe)

## Technical Approach

Two new numpy-only modules (`fitzpatrick.py`, `circle.py`) plus one new function in
`gates.py` and one in `metrics.py`. GPU work (U-Net masks, CIRCLe probe training) is
pushed behind a **file artifact seam** so every decision rule stays CPU-testable, exactly
as `metrics.py` owns its metrics instead of importing scikit-learn.

## Architecture Decisions

### D1: Fairness gate composes over `evaluate_gates`, it does not extend it

`report()` and `all_blocking_pass()` already take `Sequence[Gate]` — the composition seam
exists in the module today. `evaluate_gates` is not the registry; `list[Gate]` is.

| Option | Tradeoff | Verdict |
|---|---|---|
| **A. Standalone `fairness_gate(summary, ...) -> Gate`, caller concatenates** | Zero signature change; `gates.py` stays stdlib-only (it imports no numpy today); rollback = delete function; caller must remember to append | **Chosen** |
| B. Add `extra_gates: Sequence[Gate] = ()` param to `evaluate_gates` | One generic param, single call site, preserves ordering — but still mutates the signature and makes `evaluate_gates` a pass-through container | Rejected — buys a call site, costs the invariant |
| C. Decorator/registry of `GateProvider` callables | Global mutable state, import-order dependent; breaks under the inlined `exec`'d notebook cell; YAGNI at 7 gates | Rejected |

**Rationale**: A keeps `gates.py` a *pure decision* module — it receives numbers already
computed and decides PASS/FAIL/SKIP. The FST/ITA math never leaks into it. Call site
becomes `gates = evaluate_gates(...) + [fairness_gate(strat)]`.

### D2: Pure-numpy sRGB→LAB; `scikit-image` is a test oracle, not a runtime dep

`src/dermssl/__init__.py::MODULE_ORDER` modules are inlined and **executed** by
`test_notebook_build.py::test_core_cell_executes_and_exposes_the_full_api`. A module-level
`import skimage` in an inlined module breaks that test and the Kaggle cell.

**Choice**: ~30 lines of numpy (D65, CIE Lab). `scikit-image` goes to
`requirements-dev.txt` + `REQUIRED_FOR_FULL_RUN` **only** so a test asserts our LAB
matches `skimage.color.rgb2lab` within `atol=1e-4`. Rejected: runtime skimage (breaks
inlining, contradicts `color_constancy.py`'s "pure numpy so the transform the cluster
applies is the one tested here").

### D3: Two-tier masking, one contract

`lesion_guided.lesion_mask()` (Otsu, pure numpy, already shipped) is the **fallback**;
the ISIC-2018 U-Net is the **primary**. Both return `bool (H, W)`. Skin = `~mask`, so ITA
is computed on non-lesion pixels. Rejected: U-Net-only (no CPU test path, hard GPU
dependency for a measurement).

### D4: GPU/CPU seam = `fst_manifest.csv`

The U-Net runs once on Kaggle and emits `image_id, ita, fst_bucket, skin_fraction,
mask_source, status`. Everything downstream (stratified metrics, gate) reads that CSV.
Rejected: in-process masking during evaluation — would make the gate untestable without a GPU.

## Data Flow

    [Kaggle GPU, once]                        [CPU, every run]
    HAM/ISIC jpg                              fst_manifest.csv
        |                                          |
    torch_unet -> mask (fallback: Otsu)        join on image_id
        |                                          |
    skin = ~mask -> srgb_to_lab -> ITA        probe preds + FST bucket
        |                                          |
    fst_bucket (Groh thresholds)  ----------> metrics.stratified_summary()
        |                                          |
    fst_manifest.csv  <-----------------      gates.fairness_gate() -> Gate
                                                   |
                                         report() / all_blocking_pass()

## File Changes

| File | Action | Description |
|---|---|---|
| `src/dermssl/fitzpatrick.py` | Create | `srgb_to_lab`, `ita_degrees`, `fst_bucket`, `estimate_fst` — numpy only |
| `src/dermssl/circle.py` | Create | LAB color transformer, FST-shift sampling, blend policy (Stage 2, numpy) |
| `src/dermssl/torch_unet.py` | Create | U-Net + mask inference. GPU-only, `py_compile`/ast-tested only |
| `src/dermssl/metrics.py` | Modify | `stratified_summary(y_true, proba, buckets, n_classes)` → per-bucket dict + `gap` |
| `src/dermssl/gates.py` | Modify | Add `fairness_gate()` sibling function. `evaluate_gates` untouched |
| `src/dermssl/__init__.py` | Modify | `MODULE_ORDER += ("fitzpatrick",)` before `gates`; `circle` after |
| `src/dermssl/finetune/circle_torch.py` | Create | Joint MSE regularization loss (GPU, static-tested) |
| `scripts/notebook_code_v5.py` | Modify | Load manifest, call stratified summary, append fairness gate |
| `tests/test_notebook_build.py` | Modify | `REQUIRED_API += ["stratified_summary", "fairness_gate", "estimate_fst"]` |
| `requirements-dev.txt` | Modify | `scikit-image>=0.22` (test oracle) |
| `tests/test_environment_complete.py` | Modify | `REQUIRED_FOR_FULL_RUN += ("skimage",)` |

## Interfaces / Contracts

```python
FST_THRESHOLDS = (55.0, 41.0, 28.0, 19.0, 10.0)  # Groh et al., ITA degrees, descending

@dataclass
class FSTEstimate:
    bucket: int | None   # 1..6, None when unavailable
    ita: float           # NaN when unavailable
    skin_fraction: float
    status: str          # "ok" | "no_skin" | "degenerate_mask" | "ita_out_of_range"

def fairness_gate(
    summary: dict, *, max_gap: float = 0.15, min_bucket_n: int = 30
) -> Gate: ...
```

`fairness_gate` returns SKIP (never FAIL) when fewer than two buckets clear
`min_bucket_n` — mirroring the G2 smoke-skip precedent. Label stays Spanish:
`"G3 equidad por fototipo"`.

## Error Handling

| Condition | Behavior |
|---|---|
| Mask degenerate (area outside 1%–90%) | Fall back to Otsu; if still degenerate → `status="degenerate_mask"`, bucket `None` |
| No skin pixels (`skin_fraction < 0.05`) | `status="no_skin"`, bucket `None`, excluded from strata |
| ITA outside `[-90, 90]` | Impossible by `arctan` construction; assert as invariant, not a runtime branch |
| Bucket below `min_bucket_n` | Dropped from gap computation, counted in the gate's rendered value |
| `< 2` usable buckets | Gate = SKIP + warning (non-blocking) |
| Manifest missing | `stratified_summary` returns `{}`; gate = SKIP. v5 behavior unchanged |

## Testing Strategy

Runner: `CI_REQUIRE_FULL_DEPS=1 pytest -q -rs --strict-markers`. Strict TDD — test first.

| Layer | What | Approach |
|---|---|---|
| Unit (CPU) | LAB conversion | Property test vs. `skimage.color.rgb2lab`, `atol=1e-4` |
| Unit (CPU) | `ita_degrees`, `fst_bucket` | Synthetic swatches; boundary values at each Groh threshold |
| Unit (CPU) | `fairness_gate` | Table-driven: pass / fail / skip-thin-bucket / skip-empty |
| Unit (CPU) | `stratified_summary` | Hand-computed fixtures; NaN-safety on empty bucket |
| Unit (CPU) | `circle.py` transform | Determinism given seed; lesion pixels unchanged under mask |
| Integration (CPU) | Notebook core | Existing `exec` test + extended `REQUIRED_API` |
| Static (CPU) | `torch_unet.py`, `circle_torch.py` | `py_compile` + ast checks, per `test_pretrain_torch_layer.py` |
| Deferred (GPU) | U-Net training, mask inference, CIRCLe probe runs | Kaggle only; out of automated CI |

## Migration / Rollout

No migration. Stage 2 is gated on Stage 1 reporting a measurable gap. Absent a manifest,
the fairness gate SKIPs and v5 output is byte-identical.

## Open Questions

- [ ] ISIC 2018 Task 1 Kaggle handle unconfirmed (proposal risk, carried forward).
- [ ] `min_bucket_n = 30` is a placeholder — should be set from observed bucket counts
      after the first manifest run.
