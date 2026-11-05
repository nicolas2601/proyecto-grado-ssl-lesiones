"""Exercise real CUDA wrapper checkpoint replay using a separately sealed short smoke."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    from overlay_ssl_runtime import run
    import os
    import sys
    root = Path(__file__).resolve().parents[1]
    bindings = json.loads((root / "config/paths.json").read_text())
    sys.path.insert(0, str(Path(bindings["frozen_experiment_root"]) / "code/src"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from dermssl.pretrain import torch_resume
    config = json.loads(args.config.read_text())
    if config["base_config"]["max_steps"] != 40 or config["base_config"]["checkpoint_every_steps"] != 5:
        raise ValueError("Replay smoke requires40steps and5step checkpoints")
    out = Path(config["base_config"]["output_dir"])
    captured = out / "resume-at35.pt"
    original_save = torch_resume.atomic_save

    def capture(state, path):
        original_save(state, path)
        if state["step"] == 35:
            shutil.copyfile(path, captured)

    with patch.object(torch_resume, "atomic_save", side_effect=capture):
        straight = run(args.config)
    straight_state = torch.load(out / "resume.pt", map_location="cpu", weights_only=True)
    straight_coverage = straight["coverage"]
    resumed_config = {**config, "resume": True,
                      "base_config": {**config["base_config"], "resume_checkpoint": str(captured)}}
    resumed_path = args.config.with_name(args.config.stem + "-resume.json")
    with resumed_path.open("x") as stream:
        json.dump(resumed_config, stream, indent=2)
    resumed = run(resumed_path)
    resumed_state = torch.load(out / "resume.pt", map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "optimizer_state_dict"):
        torch.testing.assert_close(straight_state[key], resumed_state[key], rtol=0, atol=0)
    assert straight_coverage == resumed["coverage"], "Actual accessed-ID counts differ after replay"
    receipt = {"status": "PASS", "steps": 40, "resume_step": 35,
               "model_and_optimizer_bitwise_equal": True, "actual_ID_counts_equal": True,
               "training_gpu_hours": straight["allocated_gpu_hours"] + resumed["allocated_gpu_hours"]}
    with (out / "replay-smoke.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
