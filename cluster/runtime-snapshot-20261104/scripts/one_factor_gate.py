"""Single-factor guard: exactly one hyperparameter may differ from the sealed SimSiam baseline."""

ALLOWED = {'freeze_blocks': 4, 'resolution': 336}
DEFAULTS = {'freeze_blocks': 8, 'resolution': 224}
OPERATIONAL = {'output_dir', 'max_steps', 'checkpoint_every_steps', 'resume_checkpoint'}


def validate_one_factor(base, baseline):
    keys = (set(base) | set(baseline)) - OPERATIONAL
    changed = {k for k in keys if base.get(k, DEFAULTS.get(k)) != baseline.get(k, DEFAULTS.get(k))}
    if len(changed) != 1:
        raise ValueError('Exactly one factor may differ from the baseline')
    (factor,) = changed
    if factor not in ALLOWED or base.get(factor) != ALLOWED[factor]:
        raise ValueError('Factor value outside the authorized set')
    return factor
