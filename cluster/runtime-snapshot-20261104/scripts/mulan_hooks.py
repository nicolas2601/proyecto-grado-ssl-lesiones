"""MULAN local-view hook for the frozen SimSiam loop (arXiv 2602.05845 eq. 1).

Runtime-only: installed with patch.object(..., new=...); never edits the frozen source.
The frozen loop feeds only two global views, so the local crops travel through a
queue filled in the dataset and consumed in the model forward. This relies on
num_workers=0 (samples are fetched in the main process, in batch order).
"""
import copy

import numpy as np
import torch
from PIL import Image

from mulan_objective import mulan_alignment

LOCAL_QUEUE = []
LAST_STATE = {'local_predictions': None}


def _pop_local_bundles(batch_size, n_local):
    if len(LOCAL_QUEUE) < batch_size:
        raise ValueError(f'need {batch_size} local views, queue holds {len(LOCAL_QUEUE)}')
    bundles = [LOCAL_QUEUE.pop(0) for _ in range(batch_size)]
    stacked = torch.stack(bundles)
    if stacked.shape[1] != n_local:
        raise ValueError(f'expected {n_local} local views per sample, got {stacked.shape[1]}')
    return stacked


def make_mulan_simsiam(base, n_local):
    class MulanSimSiam(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # One predictor per view type: the global one is self.predictor.
            self.local_predictor = copy.deepcopy(self.predictor)
            self.n_local = n_local
            self.last_local_predictions = None

        def forward(self, view1, view2, return_features=False, tone_view=None):
            if tone_view is not None:
                raise ValueError('MULAN arm does not combine with the tone view')
            batch = view1.shape[0]
            locals_ = _pop_local_bundles(batch, self.n_local).to(view1.device)
            h1, h2 = self.backbone(view1), self.backbone(view2)
            z1, z2 = self.projector(h1), self.projector(h2)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            flat = locals_.flatten(0, 1)
            zl = self.projector(self.backbone(flat))
            q = self.local_predictor(zl).reshape(batch, self.n_local, -1)
            self.last_local_predictions = q
            LAST_STATE['local_predictions'] = q
            if return_features:
                # Same layout as the base SimSiam: index 4/5 are backbone features (lever anchors).
                return p1, p2, z1, z2, h1, h2
            return p1, p2, z1, z2
    return MulanSimSiam


def mulan_simsiam_loss(p1, p2, z1, z2, local_pred, local_weight=1.0, global_weight=1.0):
    """Symmetric SimSiam term plus MULAN local term; every local predicts the view-2 target."""
    t1, t2 = z1.detach(), z2.detach()
    global_term = 0.5 * (
        mulan_alignment({'global': p1}, t2, {'global': global_weight})
        + mulan_alignment({'global': p2}, t1, {'global': global_weight})
    )
    local_target = t2.unsqueeze(1).expand_as(local_pred)
    local_term = mulan_alignment({'local': local_pred}, local_target, {'local': local_weight})
    return global_term + local_term


def make_mulan_loss(local_weight=1.0):
    def loss(p1, p2, z1, z2):
        return mulan_simsiam_loss(p1, p2, z1, z2, LAST_STATE['local_predictions'], local_weight=local_weight)
    return loss


def make_mulan_dataset(n_local, local_resolution):
    from dermssl.pretrain.torch_data import SimSiamPairDataset, _finish_view, _geometric

    class MulanPairDataset(SimSiamPairDataset):
        def __init__(self, paths, resolution, cc_probs, seed, hue_jitter=0.1, tone_view=False):
            super().__init__(paths, resolution, cc_probs, seed, hue_jitter, tone_view)
            self.local_geometric = _geometric(local_resolution, scale=(0.05, 0.4))
            self.n_local = n_local
            self.last_locals = None

        def __getitem__(self, idx):
            view1, view2 = super().__getitem__(idx)[:2]
            rng = np.random.default_rng((self.seed, idx, 1))
            img = Image.open(self.paths[idx]).convert('RGB')
            locals_ = torch.stack([
                _finish_view(self.local_geometric(img), rng, self.cc_probs, self.hue_jitter)
                for _ in range(self.n_local)
            ])
            self.last_locals = locals_
            LOCAL_QUEUE.append(locals_)
            return view1, view2

    return MulanPairDataset
