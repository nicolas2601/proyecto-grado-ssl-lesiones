"""IConE-style runtime hooks (estilo IConE: aproximación declarada en ledger).

- Learnable instance table (N x d) registered on the model so the frozen loop's optimizer groups include it.
- The dataset queues each served index; the model forward pops one index per sample (num_workers=0 keeps order).
- Loss: icone_terms on backbone features with anchors = table[idx]; L_div uses a random subset of the table
  each step (declared deviation: full O(N^2) Gram is infeasible at N ~ 34k).
"""
import torch

from icone_losses import icone_terms

IDX_QUEUE = []
LAST = {}
N_IMAGES = [None]
DIV_SIZE = [512]


def make_icone_dataset():
    from dermssl.pretrain.torch_data import SimSiamPairDataset

    class IConEPairDataset(SimSiamPairDataset):
        def __getitem__(self, idx):
            view1, view2 = super().__getitem__(idx)[:2]
            IDX_QUEUE.append(int(idx))
            return view1, view2

    return IConEPairDataset


def _pop_indices(batch_size):
    if len(IDX_QUEUE) < batch_size:
        raise ValueError(f'need {batch_size} indices, queue holds {len(IDX_QUEUE)}')
    idx = IDX_QUEUE[:batch_size]
    del IDX_QUEUE[:batch_size]
    return torch.tensor(idx, dtype=torch.long)


def make_icone_simsiam(base):
    class IConESimSiam(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            dim = self.projector[0].in_features
            self.icone_table = torch.nn.Parameter(torch.randn(N_IMAGES[0], dim) * 0.02)

        def forward(self, view1, view2, return_features=False, tone_view=None):
            if tone_view is not None:
                raise ValueError('IConE arm does not combine with the tone view')
            idx = _pop_indices(view1.shape[0]).to(view1.device)
            h1, h2 = self.backbone(view1), self.backbone(view2)
            LAST['idx'] = idx
            LAST['table'] = self.icone_table
            if return_features:
                return h1, h2, h1.detach(), h2.detach(), h1, h2
            return h1, h2, h1, h2

    return IConESimSiam


def make_icone_loss():
    def loss(p1, p2, z1, z2):
        table = LAST['table']
        idx = LAST['idx']
        sub = torch.randint(0, table.shape[0], (min(DIV_SIZE[0], table.shape[0]),), device=table.device)
        terms = icone_terms([z1, z2], table[idx], table[sub])
        return terms['total']
    return loss
