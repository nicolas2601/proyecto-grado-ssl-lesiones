"""Adaptation levers for the SimSiam-over-DINOv2 arm.

Every lever is OFF by default, and off has to mean the v6 run exactly: the
same single optimizer group over ``model.parameters()`` and a loss that IS
the SimSiam loss. Each lever then gets a test showing it changes something,
because a flag that parses and does nothing is the failure this project has
hit most often.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from dermssl.pretrain import torch_adaptation as ad  # noqa: E402
from dermssl.pretrain.levers import Levers  # noqa: E402
from dermssl.pretrain import torch_simsiam  # noqa: E402


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    @property
    def weight(self):
        return self.fc.weight

    def forward(self, h):
        return h + self.fc(self.norm1(h))


class TinyViT(nn.Module):
    """Same attribute layout as the DINOv2 hub ViT, small enough for CPU."""

    def __init__(self, depth: int = 3, dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = dim
        self.patch_embed = nn.Linear(4, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList(TinyBlock(dim) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.patch_embed(x) + self.cls_token + self.pos_embed
        for block in self.blocks:
            h = block(h)
        return self.norm(h)


def _model(seed: int = 0) -> torch_simsiam.SimSiam:
    torch.manual_seed(seed)
    return torch_simsiam.SimSiam(backbone=TinyViT())


def _views(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(6, 4, generator=g), torch.randn(6, 4, generator=g)


class TestDefaultsAreTheV6Run:
    def test_default_levers_are_all_off(self):
        assert Levers().is_default()

    def test_one_group_over_every_parameter(self):
        model = _model()
        groups = ad.param_groups(model, Levers())
        assert len(groups) == 1
        assert [id(p) for p in groups[0]["params"]] == [id(p) for p in model.parameters()]
        assert "lr_scale" not in groups[0]

    def test_the_objective_is_the_simsiam_loss_itself(self):
        ssl = torch.tensor(-0.8)
        loss, parts = ad.compose_objective(ssl, Levers())
        assert torch.equal(loss, ssl)
        assert parts == {}

    def test_lr_is_set_unscaled_on_a_plain_group(self):
        opt = torch.optim.AdamW(_model().parameters(), lr=0.0)
        ad.apply_lr(opt, 3e-4)
        assert opt.param_groups[0]["lr"] == 3e-4

    def test_no_anchors_are_built(self):
        anchors = ad.build_anchors(_model().backbone, Levers())
        assert anchors.initial is None and anchors.teacher is None

    def test_forward_without_features_keeps_its_four_outputs(self):
        out = _model()(*_views())
        assert len(out) == 4


class TestFreeze:
    def test_embeddings_and_first_blocks_stop_training(self):
        model = _model()
        ad.freeze_blocks(model.backbone, 2)
        vit = model.backbone
        frozen = [vit.patch_embed.weight, vit.cls_token, vit.pos_embed,
                  vit.blocks[0].weight, vit.blocks[1].weight]
        assert all(not p.requires_grad for p in frozen)
        assert vit.blocks[2].weight.requires_grad and vit.norm.weight.requires_grad

    def test_layernorms_inside_frozen_blocks_keep_training(self):
        """docs/28: freeze blocks 0-7 but train EVERY LayerNorm."""
        model = _model()
        ad.freeze_blocks(model.backbone, 2)
        assert model.backbone.blocks[0].norm1.weight.requires_grad
        assert model.backbone.blocks[0].norm1.bias.requires_grad

    def test_frozen_parameters_leave_the_optimizer(self):
        model = _model()
        levers = Levers(freeze_blocks=2)
        ad.freeze_blocks(model.backbone, levers.freeze_blocks)
        params = [p for g in ad.param_groups(model, levers) for p in g["params"]]
        assert all(p.requires_grad for p in params)
        assert id(model.backbone.blocks[0].weight) not in {id(p) for p in params}

    def test_freezing_more_blocks_than_exist_refuses(self):
        with pytest.raises(ValueError):
            ad.freeze_blocks(_model().backbone, 4)


class TestLayerDecay:
    def test_scale_falls_geometrically_with_depth_from_the_top(self):
        model = _model()
        groups = ad.param_groups(model, Levers(layer_decay=0.5))
        scale_of = {id(p): g["lr_scale"] for g in groups for p in g["params"]}
        vit = model.backbone
        assert scale_of[id(vit.patch_embed.weight)] == pytest.approx(0.5**4)
        assert scale_of[id(vit.blocks[0].weight)] == pytest.approx(0.5**3)
        assert scale_of[id(vit.blocks[2].weight)] == pytest.approx(0.5**1)
        assert scale_of[id(vit.norm.weight)] == pytest.approx(1.0)
        assert scale_of[id(next(model.projector.parameters()))] == pytest.approx(1.0)

    def test_every_parameter_lands_in_exactly_one_group(self):
        model = _model()
        params = [p for g in ad.param_groups(model, Levers(layer_decay=0.7)) for p in g["params"]]
        assert sorted(id(p) for p in params) == sorted(id(p) for p in model.parameters())

    def test_apply_lr_multiplies_by_the_group_scale(self):
        model = _model()
        opt = torch.optim.AdamW(ad.param_groups(model, Levers(layer_decay=0.5)), lr=0.0)
        ad.apply_lr(opt, 1.0)
        assert sorted({g["lr"] for g in opt.param_groups}) == pytest.approx(
            sorted({0.5**k for k in range(0, 5)})
        )


class TestL2SP:
    def test_zero_at_the_starting_point(self):
        model = _model()
        anchors = ad.build_anchors(model.backbone, Levers(anchor_l2sp=1.0))
        assert ad.l2sp_penalty(model.backbone, anchors).item() == 0.0

    def test_it_is_the_squared_distance_to_the_initial_weights(self):
        model = _model()
        anchors = ad.build_anchors(model.backbone, Levers(anchor_l2sp=1.0))
        with torch.no_grad():
            model.backbone.blocks[2].weight.add_(0.1)
        expected = 0.1**2 * model.backbone.blocks[2].weight.numel()
        assert ad.l2sp_penalty(model.backbone, anchors).item() == pytest.approx(expected, rel=1e-5)

    def test_frozen_parameters_are_not_penalised(self):
        model = _model()
        anchors = ad.build_anchors(model.backbone, Levers(anchor_l2sp=1.0))
        ad.freeze_blocks(model.backbone, 1)
        with torch.no_grad():
            model.backbone.blocks[0].weight.add_(1.0)
        assert ad.l2sp_penalty(model.backbone, anchors).item() == 0.0

    def test_the_weight_enters_the_objective(self):
        ssl = torch.tensor(-0.5)
        loss, parts = ad.compose_objective(ssl, Levers(anchor_l2sp=0.1), l2sp=torch.tensor(2.0))
        assert loss.item() == pytest.approx(-0.5 + 0.2)
        assert parts["anchor_l2sp"].item() == pytest.approx(2.0)


class TestFeatureDistillation:
    def test_zero_while_the_backbone_equals_the_public_weights(self):
        model = _model()
        anchors = ad.build_anchors(model.backbone, Levers(anchor_distill=1.0))
        x, _ = _views()
        term = ad.distill_loss(model.backbone(x), anchors, x)
        assert term.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_after_drift_and_the_teacher_gets_no_gradient(self):
        model = _model()
        anchors = ad.build_anchors(model.backbone, Levers(anchor_distill=1.0))
        with torch.no_grad():
            # Random, not constant: a constant added to weights fed by a
            # LayerNorm output (zero mean) changes nothing downstream.
            g = torch.Generator().manual_seed(7)
            w = model.backbone.blocks[1].weight
            w.add_(0.5 * torch.randn(w.shape, generator=g))
        x, _ = _views()
        term = ad.distill_loss(model.backbone(x), anchors, x)
        term.backward()
        assert term.item() > 0
        assert model.backbone.blocks[1].weight.grad is not None
        assert all(p.grad is None and not p.requires_grad for p in anchors.teacher.parameters())


class TestToneConsistency:
    def test_gradient_flows_only_through_the_tone_view(self):
        h_orig = torch.randn(5, 8, requires_grad=True)
        h_tone = torch.randn(5, 8, requires_grad=True)
        ad.tone_consistency(h_orig, h_tone).backward()
        assert h_orig.grad is None
        assert h_tone.grad is not None

    def test_zero_when_the_tone_view_matches_the_original(self):
        h = torch.randn(5, 8)
        assert ad.tone_consistency(h, h.clone()).item() == pytest.approx(0.0, abs=1e-6)

    def test_it_is_added_not_multiplied(self):
        """The v6 knob rescaled the SimSiam loss; this one is a second term."""
        ssl = torch.tensor(-0.5)
        loss, parts = ad.compose_objective(ssl, Levers(tone_consistency_weight=2.0), tone=torch.tensor(0.25))
        assert loss.item() == pytest.approx(-0.5 + 0.5)
        assert parts["tone_consistency"].item() == pytest.approx(0.25)

    def test_it_uses_backbone_features_not_the_simsiam_tensors(self):
        p1, p2, z1, z2, h1, h2 = _model()(*_views(), return_features=True)
        assert h1.shape[-1] == 8 and p1.shape[-1] == torch_simsiam.PROJ_DIM

    def test_the_tone_feature_is_the_backbone_on_the_tone_view(self):
        model = _model()
        v1, v2 = _views()
        tone = 0.5 * v1
        outputs = model(v1, v2, tone_view=tone)
        assert len(outputs) == 7
        assert torch.allclose(outputs[6], model.backbone(tone))

    def test_a_lever_without_its_input_refuses(self):
        with pytest.raises(ValueError):
            ad.compose_objective(torch.tensor(0.0), Levers(tone_consistency_weight=1.0))


class TestLeverValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [{"freeze_blocks": -1}, {"layer_decay": 0.0}, {"layer_decay": 1.5},
         {"anchor_l2sp": -0.1}, {"anchor_distill": -1.0}, {"tone_consistency_weight": -1.0}],
    )
    def test_out_of_range_values_refuse(self, kwargs):
        with pytest.raises(ValueError):
            Levers(**kwargs).validate(paired=True)

    def test_tone_consistency_needs_the_paired_arm(self):
        with pytest.raises(ValueError, match="original_transformed"):
            Levers(tone_consistency_weight=1.0).validate(paired=False)


class TestLossTotalsSurviveResume:
    """Deuda P2: run.json divided post-resume sums by ALL steps (-0.047)."""

    def test_the_mean_uses_only_steps_that_were_summed(self):
        totals = ad.LossTotals()
        totals.add(ssl=-0.9, total=-0.9)
        totals.add(ssl=-0.7, total=-0.7)
        assert totals.mean("ssl") == pytest.approx(-0.8)

    def test_totals_round_trip_through_a_checkpoint(self):
        totals = ad.LossTotals()
        totals.add(ssl=-0.9, total=-0.9)
        resumed = ad.LossTotals.from_state(totals.state())
        resumed.add(ssl=-0.7, total=-0.7)
        assert resumed.mean("ssl") == pytest.approx(-0.8)

    def test_an_old_checkpoint_without_totals_starts_empty(self):
        assert ad.LossTotals.from_state(None).mean("ssl") is None


def _one_step(levers: Levers, *, freeze: bool = True):
    """Two AdamW steps through the loop's own helpers; returns the backbone weights."""
    from dermssl.pretrain import torch_loop

    model = _model(seed=3)
    anchors = ad.build_anchors(model.backbone, levers)
    if freeze:
        ad.freeze_blocks(model.backbone, levers.freeze_blocks)
    opt = torch.optim.AdamW(ad.param_groups(model, levers), lr=0.0, weight_decay=1e-4)
    ad.apply_lr(opt, 1e-2)
    need = levers.tone_consistency_weight > 0 or levers.anchor_distill > 0
    # Two steps: at step one the model sits ON its anchors, where L2-SP and
    # the feature anchor have zero gradient by construction.
    for seed in (5, 6):
        v1, v2 = _views(seed)
        if levers.tone_consistency_weight > 0:
            outputs = model(v1, v2, return_features=True, tone_view=0.9 * v1)
        else:
            outputs = model(v1, v2, return_features=True) if need else model(v1, v2)
        ssl = torch_simsiam.simsiam_loss(*outputs[:4])
        loss, _ = torch_loop._lever_objective(ssl, levers, anchors, model, outputs, v1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [p.detach().clone() for p in model.backbone.parameters()]


def _v6_step():
    """The v6 recipe written out by hand: one group, the SimSiam loss, nothing else."""
    model = _model(seed=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    for seed in (5, 6):
        v1, v2 = _views(seed)
        loss = torch_simsiam.simsiam_loss(*model(v1, v2))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [p.detach().clone() for p in model.backbone.parameters()]


class TestOneStepEndToEnd:
    def test_default_levers_update_the_weights_exactly_like_v6(self):
        for new, old in zip(_one_step(Levers()), _v6_step()):
            assert torch.equal(new, old)

    def test_the_default_objective_is_the_same_tensor(self):
        from dermssl.pretrain import torch_loop

        ssl = torch.tensor(-0.3)
        loss, parts = torch_loop._lever_objective(ssl, Levers(), ad.Anchors(), None, (), None)
        assert loss is ssl and parts == {}

    @pytest.mark.parametrize(
        "levers",
        [
            Levers(freeze_blocks=2),
            Levers(layer_decay=0.5),
            Levers(anchor_l2sp=10.0),
            Levers(anchor_distill=1.0),
            Levers(tone_consistency_weight=1.0),
        ],
        ids=["freeze", "layer_decay", "l2sp", "distill", "tone"],
    )
    def test_each_lever_changes_the_update(self, levers):
        """A flag that parses and changes nothing is the defect this guards."""
        changed = any(not torch.equal(a, b) for a, b in zip(_one_step(levers), _v6_step()))
        assert changed


class TestHeadWarmup:
    """docs/28 wave 1b (Kumar et al. 2022): heads first, backbone untouched."""

    def _step(self, model, opt, levers, step):
        from dermssl.pretrain import torch_loop

        v1, v2 = _views(step + 10)
        loss = torch_simsiam.simsiam_loss(*model(v1, v2))
        opt.zero_grad()
        loss.backward()
        torch_loop._apply_head_warmup(model, levers, step)
        opt.step()

    def test_during_warmup_the_backbone_does_not_move_at_all(self):
        model = _model(seed=4)
        levers = Levers(head_warmup_steps=2)
        opt = torch.optim.AdamW(ad.param_groups(model, levers), lr=1e-2, weight_decay=0.1)
        before = [p.detach().clone() for p in model.backbone.parameters()]
        heads_before = [p.detach().clone() for p in model.projector.parameters()]
        self._step(model, opt, levers, step=0)
        assert all(torch.equal(a, b) for a, b in zip(before, model.backbone.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(heads_before, model.projector.parameters()))

    def test_layernorms_are_frozen_too_during_warmup(self):
        model = _model(seed=4)
        levers = Levers(head_warmup_steps=1)
        opt = torch.optim.AdamW(ad.param_groups(model, levers), lr=1e-2, weight_decay=0.1)
        norm = model.backbone.blocks[0].norm1.weight.detach().clone()
        self._step(model, opt, levers, step=0)
        assert torch.equal(norm, model.backbone.blocks[0].norm1.weight)

    def test_after_warmup_the_backbone_trains(self):
        model = _model(seed=4)
        levers = Levers(head_warmup_steps=1)
        opt = torch.optim.AdamW(ad.param_groups(model, levers), lr=1e-2)
        before = [p.detach().clone() for p in model.backbone.parameters()]
        self._step(model, opt, levers, step=1)
        assert any(not torch.equal(a, b) for a, b in zip(before, model.backbone.parameters()))

    def test_zero_warmup_never_touches_gradients(self):
        from dermssl.pretrain import torch_loop

        model = _model()
        loss = torch_simsiam.simsiam_loss(*model(*_views()))
        loss.backward()
        grads = [p.grad.clone() for p in model.backbone.parameters()]
        torch_loop._apply_head_warmup(model, Levers(), step=0)
        assert all(torch.equal(g, p.grad) for g, p in zip(grads, model.backbone.parameters()))

    def test_a_negative_warmup_refuses(self):
        with pytest.raises(ValueError):
            Levers(head_warmup_steps=-1).validate(paired=False)


class TestGradCheckpoint:
    """B arms OOM at batch 128 with the third (tone) forward: recompute, same math."""

    def _grads(self, checkpoint: bool):
        model = _model()
        ad.freeze_blocks(model.backbone, 2)
        if checkpoint:
            ad.enable_grad_checkpoint(model.backbone)
        v1, v2 = _views()
        p1, p2, z1, z2 = model(v1, v2)
        torch_simsiam.simsiam_loss(p1, p2, z1, z2).backward()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    def test_gradients_match_including_frozen_block_layernorms(self):
        plain, ckpt = self._grads(False), self._grads(True)
        assert plain.keys() == ckpt.keys()
        assert "backbone.blocks.0.norm1.weight" in ckpt
        for name in plain:
            torch.testing.assert_close(ckpt[name], plain[name], atol=1e-6, rtol=0)

    def test_every_block_is_wrapped_and_state_dict_keys_unchanged(self):
        model = _model()
        keys = list(model.state_dict())
        wrapped = ad.enable_grad_checkpoint(model.backbone)
        assert wrapped == len(model.backbone.blocks)
        assert list(model.state_dict()) == keys

    def test_stochastic_blocks_refuse(self):
        model = _model()
        model.backbone.blocks[1].drop = nn.Dropout(0.1)
        with pytest.raises(ValueError, match="Dropout"):
            ad.enable_grad_checkpoint(model.backbone)
