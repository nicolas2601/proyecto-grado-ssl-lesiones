"""Budgeted launch for the approved SimSiam feature-predictor distillation arm."""
import argparse
import datetime
import json

import supervise_ssl_objective_job as supervisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    arm = 'simsiam-distill-with-predictor'
    supervisor.RUNTIMES.update({'long-dose-baseline-12000': 'longdose_ssl_runtime.py',
                               'simsiam-constant-predictor-lr': 'constant_predictor_ssl_runtime.py',
                               arm: 'predictor_distill_ssl_runtime.py'})
    supervisor.DEADLINE = datetime.datetime(2026, 10, 26, 8, 30, tzinfo=datetime.timezone.utc)
    config = json.loads((supervisor.ROOT / 'config' / (arm + '-v1/run.json')).read_text())
    remaining = (supervisor.DEADLINE - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    if remaining < config['training_gpu_hours_limit'] * 3600:
        raise ValueError('Full predictor-distillation cap cannot finish before03:30Colombia')
    supervisor.main(arm, args.gpu, args.protocol_sha)


if __name__ == '__main__':
    main()
