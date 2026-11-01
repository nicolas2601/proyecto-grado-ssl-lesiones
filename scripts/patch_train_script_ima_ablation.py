"""Derive train_dermlip_semi_026.py: the 023 trainer with a switchable IMA++ arm.

Neither existing comparison isolates IMA++. Runs 019-022 used
``train_dermlip_semi.py``, which gets copy-paste by importing
``dermssl.segmentation_methods.dermlip_slic.copy_paste``; runs 023-025 used
``train_dermlip_semi_023.py``, which carries its own 82-line
``apply_ph2_copy_paste``. So 023-vs-020 conflates IMA++ with a re-implemented
augmentation, and 023-vs-019 adds copy-paste on top of that.

This derives a single trainer where the only difference between arms is the
source of six labelled images per step:

    --ima-batch 6   10 PH2 + 6 IMA++ = 16 labelled   (reproduces 023)
    --ima-batch 0   16 PH2 + 0 IMA++ = 16 labelled   (control)

Holding the labelled batch at 16 matters: dropping the IMA++ half without
compensating would change batch size and labelled examples per step, which is a
second factor and would make the delta uninterpretable again.

A new file rather than an in-place edit, so the script hash behind the completed
023-025 runs stays intact.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PH2_LOADER_OLD = (
    "    loader_train_ph2 = DataLoader(ds_train_l, batch_size=10, shuffle=True, drop_last=True)"
)
PH2_LOADER_NEW = """    # Labelled batch is held at 16 regardless of arm: the control takes all 16 from
    # PH2, the IMA++ arm takes 10 from PH2 and 6 from the frozen cohort. Changing
    # the batch size between arms would add a second factor.
    _ima_batch = int(getattr(args, "ima_batch", 6))
    if _ima_batch < 0 or _ima_batch > 16:
        raise ValueError("--ima-batch must be between 0 and 16")
    _ph2_batch = 16 - _ima_batch
    loader_train_ph2 = DataLoader(
        ds_train_l, batch_size=_ph2_batch, shuffle=True, drop_last=True
    )"""

IMA_LOADER_OLD = """    ds_ima = IMAPlusPlusDataset(ima_manifest_path, target_size=DERMLIP_INPUT_SIZE, hair_removal=args.hair_removal)
    loader_train_ima = DataLoader(ds_ima, batch_size=6, shuffle=True, drop_last=True)"""
IMA_LOADER_NEW = """    if _ima_batch > 0:
        ds_ima = IMAPlusPlusDataset(ima_manifest_path, target_size=DERMLIP_INPUT_SIZE, hair_removal=args.hair_removal)
        loader_train_ima = DataLoader(ds_ima, batch_size=_ima_batch, shuffle=True, drop_last=True)
    else:
        ds_ima = []
        loader_train_ima = None"""

# Anchored on the preceding line: the bare assignment also occurs, more deeply
# indented, inside the StopIteration reset, and the short form is a substring of it.
ITER_OLD = """    ph2_iter = iter(loader_train_ph2)
    ima_iter = iter(loader_train_ima)"""
ITER_NEW = """    ph2_iter = iter(loader_train_ph2)
    ima_iter = iter(loader_train_ima) if loader_train_ima is not None else None"""

BATCH_OLD = """            try:
                batch_ima = next(ima_iter)
            except StopIteration:
                ima_iter = iter(loader_train_ima)
                batch_ima = next(ima_iter)"""
BATCH_NEW = """            batch_ima = None
            if ima_iter is not None:
                try:
                    batch_ima = next(ima_iter)
                except StopIteration:
                    ima_iter = iter(loader_train_ima)
                    batch_ima = next(ima_iter)"""

CAT_OLD = """            imgs_ph2, msks_ph2 = batch_ph2[0].to(device), batch_ph2[1].to(device)
            imgs_ima, msks_ima = batch_ima[0].to(device), batch_ima[1].to(device)"""
CAT_NEW = """            imgs_ph2, msks_ph2 = batch_ph2[0].to(device), batch_ph2[1].to(device)
            if batch_ima is not None:
                imgs_ima, msks_ima = batch_ima[0].to(device), batch_ima[1].to(device)
            else:
                imgs_ima = msks_ima = None"""

CONCAT_OLD = """            imgs_l = torch.cat([imgs_ph2_aug, imgs_ima], dim=0)
            msks_l = torch.cat([msks_ph2_aug, msks_ima], dim=0)"""
CONCAT_NEW = """            if imgs_ima is not None:
                imgs_l = torch.cat([imgs_ph2_aug, imgs_ima], dim=0)
                msks_l = torch.cat([msks_ph2_aug, msks_ima], dim=0)
            else:
                imgs_l, msks_l = imgs_ph2_aug, msks_ph2_aug"""

ARG_OLD = """    parser.add_argument(
        "--ima-manifest","""
ARG_NEW = """    parser.add_argument(
        "--ima-batch",
        type=int,
        default=6,
        help=(
            "Labelled images per step taken from the IMA++ frozen cohort. "
            "0 disables IMA++ and takes all 16 labelled images from PH2, which is "
            "the single-factor control arm."
        ),
    )
    parser.add_argument(
        "--ima-manifest","""

REPLACEMENTS = [
    ("ph2 loader", PH2_LOADER_OLD, PH2_LOADER_NEW),
    ("ima loader", IMA_LOADER_OLD, IMA_LOADER_NEW),
    ("ima iter", ITER_OLD, ITER_NEW),
    ("ima batch fetch", BATCH_OLD, BATCH_NEW),
    ("device move", CAT_OLD, CAT_NEW),
    ("labelled concat", CONCAT_OLD, CONCAT_NEW),
    ("cli flag", ARG_OLD, ARG_NEW),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    text = args.source.read_text(encoding="utf-8")
    print(f"source sha256: {hashlib.sha256(text.encode('utf-8')).hexdigest()}")

    for label, old, new in REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"{label}: expected exactly 1 occurrence, found {count}")
        text = text.replace(old, new)
        print(f"  patched: {label}")

    args.dest.write_text(text, encoding="utf-8")
    print(f"dest sha256:   {hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    print(f"wrote {args.dest}")


if __name__ == "__main__":
    main()
