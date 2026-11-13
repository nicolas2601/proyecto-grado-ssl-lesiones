"""Per-source label loading into one unified 8-class ontology, with modality and grouping.

Unified classes (U8): ACK, BCC, DF, MEL, NEV, SCC, SEK, VASC. The app ontology HAM7 is a projection
(akiec = ACK + SCC, bkl = SEK). Rows whose diagnosis has no U8 counterpart are dropped (None).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

W = Path("/home/jovyan/work")
U8 = ("ACK", "BCC", "DF", "MEL", "NEV", "SCC", "SEK", "VASC")
HAM7 = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
PAD6 = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
U8_TO_HAM7 = {"ACK": "akiec", "SCC": "akiec", "BCC": "bcc", "SEK": "bkl", "DF": "df", "MEL": "mel",
              "NEV": "nv", "VASC": "vasc"}
HAM7_TO_U8 = {"akiec": "ACK", "bcc": "BCC", "bkl": "SEK", "df": "DF", "mel": "MEL", "nv": "NEV", "vasc": "VASC"}
MILK_TO_U8 = {"BCC": "BCC", "NV": "NEV", "BKL": "SEK", "SCCKA": "SCC", "MEL": "MEL", "AKIEC": "ACK",
              "DF": "DF", "VASC": "VASC"}
ISIC19 = {"MEL": "MEL", "NV": "NEV", "BCC": "BCC", "AK": "ACK", "BKL": "SEK", "DF": "DF", "VASC": "VASC",
          "SCC": "SCC"}
DDI_PREFIX = (("melanoma", "MEL"), ("nodular-melanoma", "MEL"), ("basal-cell-carcinoma", "BCC"),
              ("squamous-cell-carcinoma", "SCC"), ("actinic-keratosis", "ACK"), ("seborrheic-keratosis", "SEK"),
              ("dermatofibroma", "DF"), ("pyogenic-granuloma", "VASC"), ("angioma", "VASC"),
              ("arteriovenous-hemangioma", "VASC"), ("melanocytic-nevi", "NEV"), ("dysplastic-nevus", "NEV"),
              ("blue-nevus", "NEV"))
INTERNET = {"melanoma": "MEL", "bcc": "BCC", "scc": "SCC", "actinic_keratosis": "ACK",
            "seborrheic_keratosis": "SEK", "nevus": "NEV", "dermatofibroma": "DF", "vascular": "VASC"}


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def ham():
    out = {}
    with (W / "datos/ham10000/HAM10000_metadata.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            out[r["image_id"]] = (HAM7_TO_U8[r["dx"]], "derm", r["lesion_id"])
    return out


def isic2019():
    root = W / "datos/isic2019_fullres"
    lesion = {}
    with (root / "ISIC_2019_Training_Metadata.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            lesion[r["image"]] = r["lesion_id"] or r["image"]
    out = {}
    with (root / "ISIC_2019_Training_GroundTruth.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            hit = [k for k in ISIC19 if float(r[k]) == 1.0]
            if len(hit) == 1:
                out[r["image"]] = (ISIC19[hit[0]], "derm", lesion.get(r["image"], r["image"]))
    return out


def pad():
    out = {}
    with (W / "pad_ufes_20/metadata.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            out[Path(r["img_id"]).stem] = (r["diagnostic"], "clin", r["patient_id"])
    return out


def milk():
    out = {}
    with (W / "milk10k/manifest-clean.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            label = MILK_TO_U8.get(r["milk_class"])
            if label:
                modality = "clin" if r["modality"] == "clinica" else "derm"
                out[r["image_id"]] = (label, modality, r["lesion_id"])
    return out


def ddi():
    out = {}
    for r in _jsonl(W / "ddi-20261105/ddi_holdout.jsonl"):
        label = next((u for prefix, u in DDI_PREFIX if r["disease"].startswith(prefix)), None)
        if label:
            out[r["image_id"]] = (label, "clin", r["identity_component_id"])
    return out


def bosque():
    return {r["image_id"]: (HAM7_TO_U8[r["histology_label"]], "derm", r["identity_component_id"])
            for r in _jsonl(W / "bosque-20261108/bosque_colombia_holdout.jsonl")
            if r["histology_label"] in HAM7_TO_U8}


def fitz(path=W / "nevus-fix-20261113/fitz17k/fitz_rows.jsonl"):
    return {r["image_id"]: (r["label"], "clin", r["image_id"]) for r in _jsonl(path)}


def internet(path=W / "nevus-fix-20261113/internet/labels.csv"):
    out = {}
    with Path(path).open(newline="") as stream:
        for r in csv.DictReader(stream):
            label = INTERNET.get(r["label"])
            if label:
                out[Path(r["file"]).stem] = (label, "clin" if r["modality"] == "clinical" else "derm", r["file"])
    return out


SOURCES = {"ham_fit": ham, "ham_dev": ham, "ham_holdout": ham, "isic2019_noham": isic2019,
           "pad_adapt": pad, "pad_test": pad, "milk": milk, "ddi": ddi, "bosque": bosque, "fitz": fitz,
           "internet": internet}
