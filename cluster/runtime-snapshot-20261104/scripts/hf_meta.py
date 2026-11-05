import json, urllib.request
out = {}
for repo in ["facebook/dinov2-small", "facebook/dinov2-giant"]:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=30) as f:
        d = json.load(f)
    files = {s["rfilename"]: s for s in d.get("siblings", [])}
    rec = {"repo_id": repo, "revision": d["sha"], "private": d["private"], "gated": d["gated"],
           "license_tag": "license:apache-2.0" in d.get("tags", []), "files": []}
    for name in ["config.json", "preprocessor_config.json", "README.md", "model.safetensors"]:
        s = files.get(name)
        if s is None:
            rec["files"].append({"name": name, "missing": True}); continue
        lfs = s.get("lfs") or {}
        rec["files"].append({"name": name, "bytes": s.get("size"), "published_lfs_sha256": lfs.get("sha256"), "git_blob_sha1": s.get("blobId")})
    out[repo] = rec
    print(repo, "rev", d["sha"][:12], "gated", d["gated"], "license", rec["license_tag"], "has_safetensors", "model.safetensors" in files, "open_clip_files", [k for k in files if "open_clip" in k][:2])
json.dump(out, open("/tmp/dino_meta.json", "w"), indent=1)
