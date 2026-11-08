import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/build_moco_bank_comparison.py"
spec = importlib.util.spec_from_file_location("moco_bank", path)
bank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bank)


def test_bank_rows_remain_separate_and_keep_training_protocol_labels():
    data = {"bank-x-finetune": {"PAD": {"macro_f1": .5,
            "per_class": {"a": {"precision": .4, "recall": .5, "f1": .44}}}}}
    rows = bank.rows(data)
    assert len(rows) == 1
    assert rows[0]["protocol"] == "supervised fine-tuning bank; not matched frozen LR"
    assert rows[0]["class"] == "a"
    assert rows[0]["f1"] == .44


def test_bank_does_not_relabel_other_families_as_supervised_baselines():
    assert bank.rows({"MoCo-v3-style": {}, "SimSiam (reference)": {}}) == []
    with pytest.raises(ValueError):
        bank.rows({"bank-x-frozen": {"PAD": {"per_class": {"a": {"precision": 2, "recall": .5, "f1": .5}}}}})
