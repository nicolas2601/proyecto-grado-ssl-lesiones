"""Runner v2 (per-arm script field). Detached, resumable runner for SSL arms: smoke 40 -> replay SHA -> full 6000 -> readout -> stats -> registry -> table.

Queue: results/campaign-20261027-v3/runner/queue.json (re-read every iteration).
Status: queue.json (per arm) and STATUS.json (summary). Pause: create runner/PAUSE. Failures are recorded and skipped.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/jovyan/work/clf1_clinical_reference_20261024')
RUN = ROOT / 'results/campaign-20261027-v3/runner'
QUEUE = RUN / 'queue.json'
STATUS = RUN / 'STATUS.json'
LOGS = RUN / 'logs'
PY = '/home/jovyan/work/proyectonp/.venv/bin/python'
REG = ROOT / 'results/campaign-20261027-v3/registro-union.json'
ALLOWED_SEEDS = {42, 43, 44}  # guardia de semillas: NO ampliar
STEP_TIMEOUT = {'smoke': 3600, 'replay': 3600, 'full': 21600, 'readout': 1800, 'stats': 900}


DEFAULT_SCRIPT = {'projector': 'scripts/projector_ssl_runtime.py', 'overlay': 'scripts/run_overlay_ssl.py'}


def resolve_script(arm):
    """Per-arm runtime: the arm's 'script' field wins; otherwise the default for its tool."""
    return arm.get('script') or DEFAULT_SCRIPT[arm['tool']]


def promote_holds(q):
    """Arms on hold become pending and go to the END of the queue; the order of the other arms is unchanged."""
    rest = [a for a in q if a.get('status') != 'hold']
    held = [a for a in q if a.get('status') == 'hold']
    for a in held:
        a['status'] = 'pending'
    return rest + held


def log(msg):
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / 'runner.log', 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def gpu_free():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.split('\n')
    free = [int(l.split(',')[0]) for l in out if l.strip() and int(l.split(',')[1]) < 1000]
    return str(free[0]) if free else None


def mem_ok():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) / 1024 / 1024 > 2.0  # > 2 GiB
    return True


def run(cmd, env, timeout, logfile):
    with open(logfile, 'w') as f:
        p = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode


def sha_of(path):
    return json.loads(Path(path).read_text())['backbone_sha256']


def load_queue():
    return json.loads(QUEUE.read_text())


def save_queue(q, arm_id=None):
    """Merge: re-read the file, keep every entry, and update only the status/error of arm_id."""
    fresh = load_queue()
    mine = {a['id']: a for a in q}
    for a in fresh:
        if arm_id and a['id'] == arm_id and a['id'] in mine:
            a['status'], a['error'] = mine[a['id']]['status'], mine[a['id']].get('error')
    tmp = QUEUE.with_suffix('.tmp')
    tmp.write_text(json.dumps(fresh, indent=2, ensure_ascii=False))
    os.replace(tmp, QUEUE)


def write_status(q):
    counts = {}
    for a in q:
        counts[a['status']] = counts.get(a['status'], 0) + 1
    STATUS.write_text(json.dumps({'updated': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'counts': counts,
                                  'arms': [{'id': a['id'], 'status': a['status'], 'error': a.get('error')} for a in q]},
                                 indent=2, ensure_ascii=False))


def wait_full(arm):
    target = ROOT / arm['full_out'] / 'result.json'
    while True:
        if target.exists() and json.loads(target.read_text()).get('status') == 'COMPLETE':
            return
        time.sleep(60)


def arm_steps(arm, env):
    cfg = ROOT / arm['config_dir']
    seed = arm['seed']
    if seed not in ALLOWED_SEEDS:
        raise ValueError('seed outside authorized guard')
    tool = arm['tool']  # 'projector' or 'overlay'
    script = resolve_script(arm)
    if arm.get('skip_train'):
        wait_full(arm)
        steps = ()
    else:
        steps = ('smoke', 'replay', 'full')
    for step in steps:
        name = 'run.json' if step == 'full' else f'{step}.json'
        conf = cfg / name
        rc = run([PY, script, str(conf)], env, STEP_TIMEOUT[step], LOGS / f"{arm['id']}-{step}.log")
        if rc != 0:
            raise RuntimeError(f'{step} exit {rc}')
        if step == 'replay':
            s = sha_of(ROOT / arm['smoke_out'] / 'result.json')
            r = sha_of(ROOT / arm['replay_out'] / 'result.json')
            if s != r:
                raise RuntimeError('replay SHA mismatch')
    weights = ROOT / arm['full_out'] / 'simsiam_dinov2_backbone_final.pt'
    out = ROOT / f"results/{arm['id']}-evaluation-v1"
    out.mkdir(parents=True, exist_ok=True)
    R = ROOT / 'results/content-exclusion-overlay-v1'
    import hashlib
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    rs, ps, hs, ws = h(R / 'receipt.json'), h(R / 'pad_adapt.jsonl'), h(R / 'ham_dev.jsonl'), h(weights)
    pad_npz, ham_npz = out / 'pad_adapt.npz', out / 'ham_dev.npz'
    rc = run([PY, 'scripts/export_overlay_features.py', '--role', 'pad_adapt', '--receipt', str(R / 'receipt.json'), '--receipt-sha', rs,
              '--manifest', str(R / 'pad_adapt.jsonl'), '--manifest-sha', ps, '--weights', str(weights), '--weights-sha', ws,
              '--output', str(pad_npz)], env, STEP_TIMEOUT['readout'], LOGS / f"{arm['id']}-export-pad.log")
    rc |= run([PY, 'scripts/export_overlay_features.py', '--role', 'ham_dev', '--receipt', str(R / 'receipt.json'), '--receipt-sha', rs,
               '--manifest', str(R / 'ham_dev.jsonl'), '--manifest-sha', hs, '--weights', str(weights), '--weights-sha', ws,
               '--output', str(ham_npz)], env, STEP_TIMEOUT['readout'], LOGS / f"{arm['id']}-export-ham.log")
    rc |= run([PY, 'scripts/evaluate_overlay_readout.py', '--kind', 'PAD', '--features', str(pad_npz), '--manifest', str(R / 'pad_adapt.jsonl'),
               '--metadata', '/home/jovyan/work/pad_ufes_20/metadata.csv', '--receipt', str(R / 'receipt.json'), '--output', str(out / 'PAD-readout.json'),
               '--metadata-sha', '14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527', '--receipt-sha', rs],
              env, STEP_TIMEOUT['readout'], LOGS / f"{arm['id']}-readout-pad.log")
    rc |= run([PY, 'scripts/evaluate_overlay_readout.py', '--kind', 'HAM', '--features', str(ham_npz), '--manifest', str(R / 'ham_dev.jsonl'),
               '--metadata', '/home/jovyan/work/datos/ham10000/HAM10000_metadata.csv', '--receipt', str(R / 'receipt.json'),
               '--output', str(out / 'HAM-readout-06aee5.json'),
               '--metadata-sha', '06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5', '--receipt-sha', rs],
              env, STEP_TIMEOUT['readout'], LOGS / f"{arm['id']}-readout-ham.log")
    if rc:
        raise RuntimeError('readout failed')
    stats = run_stats(out, env, arm)
    append_registry(arm, stats)
    render_table()


def run_stats(out, env, arm):
    res = subprocess.run([PY, 'scripts/arm_stats.py', str(out)], env=env, capture_output=True, text=True,
                         timeout=STEP_TIMEOUT['stats'])
    if res.returncode != 0:
        raise RuntimeError('stats failed: ' + res.stderr[-300:])
    (out / 'stats.json').write_text(res.stdout)
    return json.loads(res.stdout)


def append_registry(arm, st):
    shutil.copy(REG, REG.with_suffix('.bak-runner.json'))
    data = json.loads(REG.read_text())
    p, h = st['PAD'], st['HAM']
    row = {'name': arm['label'], 'internal': arm['id'], 'section': arm['section'], 'type': arm['type'],
           'seeds': f"SSL 1 / CV 3", 'pad_macro_f1': p['mean_f1'], 'delta_pad': [round(p['dmean'], 4), p['ic'][0], p['ic'][1]],
           'delta_pad_mean': round(p['dseed'], 4), 'ham_macro_f1': h['mean_f1'],
           'delta_ham': [round(h['dmean'], 4), h['ic'][0], h['ic'][1]], 'delta_ham_mean': round(h['dseed'], 4),
           'auc_cancer_mean': round(st['auc_cancer'], 4), 'recall_mean': {k: round(v, 4) for k, v in st['recall'].items()},
           'gpu_h': None, 'status': arm.get('status_text', 'exploratorio (runner)')}
    data['rows'] = [r for r in data['rows'] if r['internal'] != arm['id']] + [row]
    tmp = REG.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, REG)


def render_table():
    subprocess.run([PY, '-c', 'from render_results_table import main; '
                    'main("results/campaign-20261027-v3/registro-union.json", "results/campaign-20261027-v3/tabla-maestra-union")'],
                   env={**os.environ, 'PYTHONPATH': 'scripts'}, cwd=ROOT, timeout=300)


def main():
    q0 = load_queue()
    q1 = promote_holds(q0)
    if q1 != q0:
        tmp = QUEUE.with_suffix('.tmp')
        tmp.write_text(json.dumps(q1, indent=2, ensure_ascii=False))
        os.replace(tmp, QUEUE)
        log('v2 startup: promoted hold arms to end of queue')
    while True:
        if (RUN / 'PAUSE').exists():
            time.sleep(60)
            continue
        q = load_queue()
        pending = [a for a in q if a['status'] == 'pending']
        if not pending:
            write_status(q)
            time.sleep(120)
            continue
        arm = pending[0]
        if not mem_ok():
            time.sleep(120)
            continue
        gpu = gpu_free()
        if gpu is None:
            time.sleep(120)
            continue
        arm['status'] = 'running'
        save_queue(q, arm['id'])
        write_status(load_queue())
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'PYTHONPATH': 'scripts:' + str(ROOT / 'scripts'),
               'OMP_NUM_THREADS': '2'}
        try:
            arm_steps(arm, env)
            arm['status'] = 'done'
            arm.pop('error', None)
        except Exception as e:  # keep the queue going
            arm['status'] = 'FAILED'
            arm['error'] = str(e)[:300]
            log(f"{arm['id']} FAILED: {e}")
        save_queue(q, arm['id'])
        write_status(load_queue())
        log(f"{arm['id']} -> {arm['status']}")


if __name__ == '__main__':
    main()
