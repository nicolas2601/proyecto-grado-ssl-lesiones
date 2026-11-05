"""Launch only the approved single-GPU SimSiam constant-predictor-LR ablation."""
import argparse
import datetime
import json

import supervise_ssl_objective_job as supervisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    arm = 'simsiam-constant-predictor-lr'
    supervisor.RUNTIMES.update({'long-dose-baseline-12000': 'longdose_ssl_runtime.py',
                               arm: 'constant_predictor_ssl_runtime.py'})
    config = json.loads((supervisor.ROOT / 'config' / (arm + '-v1/run.json')).read_text())
    remaining = (supervisor.DEADLINE - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    if remaining < config['training_gpu_hours_limit'] * 3600:
        raise ValueError('Full predictor ablation cap cannot finish within the overnight window')
    supervisor.main(arm, args.gpu, args.protocol_sha)


if __name__ == '__main__':
    main()
