"""Local-only original/pseudo overlays for ordered visual checking, never W&B media."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def save_review(rgb_chw,pseudos,valid,directory):
    rgb=np.asarray(rgb_chw).transpose(1,2,0)
    pseudos=np.asarray(pseudos);valid=np.asarray(valid).astype(bool)
    if rgb.shape[:2]!=valid.shape or pseudos.ndim!=3 or pseudos.shape[1:]!=valid.shape:
        raise ValueError('Overlay geometry must match the transformed original')
    if not np.isfinite(rgb).all() or not np.isfinite(pseudos).all() or ((pseudos<0)|(pseudos>1)).any():
        raise ValueError('Finite aligned overlay inputs required')
    directory=Path(directory)
    Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).save(directory/'original_resized.png')
    for i,pseudo in enumerate(pseudos):
        lesion=(pseudo>=.65)&valid;uncertain=((pseudo>.35)&(pseudo<.65))|~valid
        color=np.zeros_like(rgb);color[lesion]=np.array([216,120,81])/255;color[uncertain]=np.array([185,145,60])/255
        visible=lesion|uncertain;overlay=rgb.copy();overlay[visible]=.65*rgb[visible]+.35*color[visible]
        Image.fromarray(np.rint(np.clip(overlay,0,1)*255).astype(np.uint8)).save(directory/f'overlay_{i}.png')
    (directory/'legend.json').write_text(json.dumps({'foreground':'pseudo only, orange','uncertain_or_invalid':'gold',
        'not_expert_truth':True,'upload_to_wandb':False},indent=2)+'\n')
