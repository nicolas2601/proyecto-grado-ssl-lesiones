"""DIET-CP-style objective (estilo DIET-CP: aproximación declarada en ledger).

Each training image is its own class (its index). A linear head over the CLS token predicts that index with
label smoothing 0.3. Warm-up (first 5 % of steps): only the head trains. Afterwards the last two blocks train.
"""
import torch.nn.functional as F


def diet_loss(logits, idx, smoothing=0.3):
    return F.cross_entropy(logits, idx, label_smoothing=smoothing)


def backbone_trainable(step, total_steps, blocks, warmup_fraction=0.05, last=2):
    if step <= warmup_fraction * total_steps:
        return set()
    return set(range(blocks - last, blocks))
