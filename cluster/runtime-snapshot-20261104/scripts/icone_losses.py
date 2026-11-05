"""IConE-style objective (estilo IConE: aproximación declarada en ledger).

Terms follow the published decomposition: L = L_vi + L_vv + L_div, unweighted.
- L_vi: view-to-anchor alignment (gradients to encoder and anchor table).
- L_vv: view-to-view consistency (gradients to encoder only).
- L_div: squared hinge on positive anchor correlations over the full table (gradients to table only).
Anchors are a learnable N x d table indexed by instance id; the table is used only in training.
"""
import torch
import torch.nn.functional as F


def icone_terms(views, anchors, table):
    """views: list of V tensors (B, d) of encoder outputs; anchors: (B, d) rows of the table; table: (N, d)."""
    zs = [F.normalize(v, dim=-1) for v in views]
    e_hat_b = F.normalize(anchors, dim=-1)
    v_count = len(zs)
    vi = torch.stack([(1 - (z * e_hat_b).sum(-1)).mean() for z in zs]).mean()
    pairs = []
    for m in range(v_count):
        for n in range(m + 1, v_count):
            pairs.append((1 - (zs[m] * zs[n]).sum(-1)).mean())
    vv = torch.stack(pairs).mean() if pairs else torch.zeros((), device=anchors.device)
    e_hat = F.normalize(table, dim=-1)
    gram = e_hat @ e_hat.T
    n = table.shape[0]
    off = gram - torch.diag(torch.diagonal(gram))
    div = off.clamp(min=0).pow(2).sum() / (n * (n - 1))
    total = vi + vv + div
    return {'vi': vi, 'vv': vv, 'div': div, 'total': total}
