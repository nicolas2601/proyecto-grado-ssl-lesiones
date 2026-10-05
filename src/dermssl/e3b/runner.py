"""E3b end to end: descriptors, B71 references, then each backbone (extract, score), then the report.

Order matters: the descriptors and the B71 reference cell run FIRST, because the B71 arms are
checked against ``e3_final_eval.json`` and a mismatch means the pairing is broken; the run
stops there instead of spending GPU hours on eight backbones. The backbone weights are
loaded one model at a time and released before the next.
"""

from __future__ import annotations

import gc

from dermssl import e3_store as st
from dermssl.e3b import pipeline, report
from dermssl.e3b.runtime import Runtime, Settings


def _finish(cells: st.CellStore, rt: Runtime, settings: Settings, sidecar, models: list[str]) -> int:
    for name, error in cells.failures:
        print(f"[FALLO] {name}: {error}")
    summary = report.write_outputs(cells, rt, settings, models, cells.failures, sidecar)
    print(f"[e3b] complete={summary['complete']} -> {settings.out_dir / report.SUMMARY_FILE}")
    return 0 if summary["complete"] else 1


class _Lazy:
    """The backbone of one model, built on first use and dropped afterwards."""

    def __init__(self, build):
        self.build, self.value = build, None

    def __call__(self):
        if self.value is None:
            self.value = self.build()
        return self.value

    def release(self) -> None:
        self.value = None
        gc.collect()
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run(rt: Runtime, settings: Settings, sidecar) -> int:
    """Run (or resume) every cell; the exit code is 1 when any cell failed or the summary is incomplete."""
    cells = st.CellStore(settings.out_dir / "cells")
    models = list(settings.models)
    pipeline.descriptor_cells(cells, rt, settings, sidecar)
    if not cells.failures:
        report.b71_cell(cells, rt, settings, sidecar)
    if cells.failures:
        print("[ABORT] no se pudo fijar la referencia B71 ni los descriptores: no se extrae ningun baseline")
        return _finish(cells, rt, settings, sidecar, models)
    for model in models:
        _one_model(cells, rt, settings, model, sidecar)
    return _finish(cells, rt, settings, sidecar, models)


def _one_model(cells: st.CellStore, rt: Runtime, settings: Settings, model: str, sidecar) -> None:
    lazy = _Lazy(lambda: rt.build_backbone(model))
    if pipeline.extract_cells(cells, rt, settings, model, lazy, sidecar):
        pipeline.eval_cell(cells, rt, settings, model, sidecar)
    lazy.release()
