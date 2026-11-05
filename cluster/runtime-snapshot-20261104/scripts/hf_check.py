import json, urllib.request
repos = ["facebook/dinov2-small", "facebook/dinov2-giant", "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
         "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
         "suinleelab/monet", "redlessone/DermLIP_ViT-B-16", "facebook/dinov3-vitb16-pretrain-lvd1689m", "google/medsiglip-448"]
for r in repos:
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{r}", timeout=20) as f:
            d = json.load(f)
        tags = d.get("tags", [])
        lic = [t for t in tags if t.startswith("license:")]
        print(r, "| sha", d.get("sha", "?")[:12], "| gated", d.get("gated"), "| private", d.get("private"), "|", lic, "| remote_code", any("custom_code" in t for t in tags))
    except Exception as e:
        print(r, "| ERROR", type(e).__name__, str(e)[:80])
