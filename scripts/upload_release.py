"""Upload release/v7 to a GitHub Release and/or the Hugging Face Hub. Dry run by default.

The token is read from a file (``--token-file`` or ``$RELEASE_TOKEN_FILE``)
that must be private (mode 600); it is passed to ``gh`` through the
environment and to ``huggingface_hub`` as an argument, and never printed.
Before anything is sent, every file is checked against SHA256SUMS and against
the 2 GiB GitHub asset limit. GitHub assets are flat, so ``B_s42/onnx/model.onnx``
is uploaded as ``B_s42__onnx__model.onnx``; the Hub keeps the folder layout.

    python scripts/upload_release.py --target github --github-repo OWNER/REPO \\
        --token-file ~/work/proyectonp/.secrets/github_token            # dry run
    python scripts/upload_release.py --target hf --hf-repo OWNER/v7-dermoscopy \\
        --token-file ~/work/proyectonp/.secrets/hf_token --execute
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl.release import package  # noqa: E402

TAG = "v7.0.0"
TITLE = "v7 skin-lesion classifiers (research release)"


class UploadRefused(RuntimeError):
    """Something about the release or the token is not safe to upload."""


def read_token(path) -> str:
    path = Path(path).expanduser()
    if not path.is_file():
        raise UploadRefused(f"token file {path} does not exist")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise UploadRefused(f"token file {path} is readable by others; chmod 600 it")
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise UploadRefused(f"token file {path} is empty")
    return token


def verify_checksums(root: Path) -> int:
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        raise UploadRefused("SHA256SUMS missing; run scripts/finalize_release.py")
    listed = [line.split("  ", 1) for line in sums.read_text(encoding="utf-8").splitlines() if line]
    for digest, rel in listed:
        if package.sha256_file(root / rel) != digest:
            raise UploadRefused(f"{rel} does not match SHA256SUMS")
    unlisted = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    unlisted -= {rel for _, rel in listed} | {"SHA256SUMS"}
    if unlisted:
        raise UploadRefused(f"files not in SHA256SUMS: {sorted(unlisted)}")
    return len(listed)


def github_assets(root: Path, limit_bytes: int = package.GITHUB_ASSET_LIMIT) -> list:
    """``(path, asset_name)`` for every file; refuses oversized files and name clashes."""
    try:
        package.size_report(root, limit_bytes)
    except package.AssetTooLarge as exc:
        raise UploadRefused(str(exc)) from exc
    assets = [(p, p.relative_to(root).as_posix().replace("/", "__"))
              for p in sorted(root.rglob("*")) if p.is_file()]
    names = [name for _, name in assets]
    if len(set(names)) != len(names):
        raise UploadRefused("two files flatten to the same asset name")
    return assets


def stage_github(assets: list, staging: Path) -> list:
    staging.mkdir(parents=True, exist_ok=True)
    staged = []
    for path, name in assets:
        link = staging / name
        link.unlink(missing_ok=True)
        link.symlink_to(path.resolve())
        staged.append(link)
    return staged


def github_command(tag: str, repo: str, notes: Path, staged: list) -> list:
    return ["gh", "release", "create", tag, "--repo", repo, "--title", TITLE,
            "--notes-file", str(notes), *map(str, staged)]


def upload_github(root: Path, args, token: str | None, runner=subprocess.run) -> list:
    assets = github_assets(root)
    staged = stage_github(assets, args.staging)
    command = github_command(args.tag, args.github_repo, root / "README.md", staged)
    print(f"github: {len(staged)} assets -> {args.github_repo} {args.tag}")
    if args.execute:
        runner(command, check=True, env={**os.environ, "GH_TOKEN": token})
    else:
        print("dry run:", " ".join(command[:9]), f"... ({len(staged)} files)")
    return command


def upload_hf(root: Path, args, token: str | None, api_factory=None) -> dict:
    plan = {"repo_id": args.hf_repo, "files": sum(1 for p in root.rglob("*") if p.is_file())}
    print(f"hf: {plan['files']} files -> {args.hf_repo} (model repo)")
    if not args.execute:
        print("dry run: nothing sent")
        return plan
    if api_factory is None:
        from huggingface_hub import HfApi as api_factory  # noqa: PLC0415, N813
    api = api_factory(token=token)
    api.create_repo(repo_id=args.hf_repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(root), repo_id=args.hf_repo, repo_type="model",
                      commit_message=f"Add {args.tag} release")
    return plan


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--release", type=Path, default=REPO / "release" / "v7")
    parser.add_argument("--target", choices=("github", "hf"), required=True)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--github-repo")
    parser.add_argument("--hf-repo")
    parser.add_argument("--token-file", default=os.environ.get("RELEASE_TOKEN_FILE"))
    parser.add_argument("--staging", type=Path, default=REPO / "release" / "github_assets")
    parser.add_argument("--execute", action="store_true", help="actually upload (default: dry run)")
    args = parser.parse_args(argv)
    if args.target == "github" and not args.github_repo:
        parser.error("--github-repo is required for --target github")
    if args.target == "hf" and not args.hf_repo:
        parser.error("--hf-repo is required for --target hf")
    if args.execute and not args.token_file:
        parser.error("--execute needs --token-file or RELEASE_TOKEN_FILE")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        token = read_token(args.token_file) if args.token_file else None  # validated, never printed
        print(f"checksums OK: {verify_checksums(args.release)} files")
        if args.target == "github":
            upload_github(args.release, args, token)
        else:
            upload_hf(args.release, args, token)
    except UploadRefused as exc:
        print(f"ABORT: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
