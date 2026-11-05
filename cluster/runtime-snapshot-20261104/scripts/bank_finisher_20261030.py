"""Wait for both bank queues, then write full-metrics.json for every bank job and the SSL references."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results/baseline-bank-20261030"
PYTHON = "/home/jovyan/work/proyectonp/.venv/bin/python"
SSL = ["SimSiam (referencia)=baseline-s42-pipeline-evaluation-v1,baseline-s43-evaluation-v1,baseline-s44-evaluation-v1",
       "VICReg=vicreg-evaluation-v1,vicreg-s43-evaluation-v1,vicreg-s44-evaluation-v1",
       "Barlow + VICReg=barlow-vicreg-evaluation-v1,barlow-vicreg-s43-evaluation-v1,barlow-vicreg-s44-evaluation-v1",
       "MoCo-v3-style=moco-v3-s42-evaluation-v1,moco-v3-s43-evaluation-v1,moco-v3-s44-evaluation-v1",
       "DINO + iBOT=dino-ibot-evaluation-v1,dino-ibot-s43-evaluation-v1,dino-ibot-s44-evaluation-v1"]


def main():
    markers = [STATE / f"gpu{g}/ALL_TERMINAL.json" for g in (0, 1)]
    while not all(m.exists() for m in markers):
        time.sleep(60)
    jobs = [j for m in markers for j in json.loads(m.read_text())["jobs"]]
    done = [f"{j['id']}={j['id']}-evaluation-v1" for j in jobs if j["status"] == "done"]
    subprocess.run([PYTHON, "scripts/full_metrics_20261030.py", str(STATE / "full-metrics.json"), *SSL, *done],
                   cwd=ROOT, env={"PYTHONPATH": "scripts", "PATH": "/usr/bin:/bin"}, timeout=1800, capture_output=True)
    (STATE / "BANK_DONE.json").write_text(json.dumps({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                      "jobs": len(jobs), "done": len(done)}) + "\n")


if __name__ == "__main__":
    main()
