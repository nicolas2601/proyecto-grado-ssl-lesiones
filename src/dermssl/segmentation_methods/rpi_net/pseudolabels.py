"""Four-source pseudo construction from reviewed SSL features, never diagnoses."""
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans
from dermssl.w3_components import DegenerateCase


def cluster_foreground(features, output_size, *, seed=42, n_init=10):
    """Two clusters; border-majority is background (declared missing-paper choice)."""
    features=np.asarray(features,dtype=np.float64)
    if features.ndim!=3 or not np.isfinite(features).all() or min(features.shape[:2])<2:
        raise ValueError('Finite HWC feature grid required')
    flat=features.reshape(-1,features.shape[-1])
    if np.unique(flat,axis=0).shape[0]<2: raise DegenerateCase('Constant SSL feature grid')
    labels=KMeans(n_clusters=2,n_init=n_init,random_state=seed).fit_predict(flat).reshape(features.shape[:2])
    border=np.concatenate((labels[0],labels[-1],labels[1:-1,0],labels[1:-1,-1]))
    counts=np.bincount(border,minlength=2)
    if counts[0]==counts[1]: raise DegenerateCase('Ambiguous cluster polarity at border')
    foreground=(labels!=int(np.argmax(counts))).astype(np.float32)
    return F.interpolate(torch.from_numpy(foreground)[None,None],size=output_size,mode='nearest')[0,0].numpy()


def gradcampp_energy(activations, output_size):
    """Positive-gradient-power GradCAM++ with class-agnostic SSL energy target.

    The paper does not specify a classifier target for an SSL backbone. Choosing
    mean squared feature energy is OUR adaptation, not an inferred diagnosis.
    """
    if activations.ndim!=4 or not activations.requires_grad or not torch.isfinite(activations).all():
        raise ValueError('Finite differentiable BCHW activations required')
    energy=activations.square().mean((-3,-2,-1)).sum()
    grad=torch.autograd.grad(energy,activations,retain_graph=True)[0]
    square=grad.square(); cube=square*grad
    denominator=2*square+activations.sum((-2,-1),keepdim=True)*cube
    alpha=torch.where(square>0,square/denominator.clamp_min(1e-8),torch.zeros_like(square))
    weights=(alpha*grad.relu()).sum((-2,-1),keepdim=True)
    cam=(weights*activations).sum(1,keepdim=True).relu()
    cam=F.interpolate(cam,size=output_size,mode='bilinear',align_corners=False)
    low=cam.amin((-2,-1),keepdim=True); span=cam.amax((-2,-1),keepdim=True)-low
    if (span<=1e-12).any(): raise DegenerateCase('Weak or constant GradCAM++ energy map')
    return ((cam-low)/span).detach()


def four_sources(mocov3_conv1, mocov3_cam, mocov2_conv1, dino_final, output_size=(224,224), seed=42):
    cam=np.asarray(mocov3_cam,dtype=np.float32)
    if cam.shape!=output_size or not np.isfinite(cam).all() or ((cam<0)|(cam>1)).any():
        raise ValueError('Aligned foreground CAM in [0,1] required')
    return np.stack((cluster_foreground(mocov3_conv1,output_size,seed=seed),cam,
                     cluster_foreground(mocov2_conv1,output_size,seed=seed),
                     cluster_foreground(dino_final,output_size,seed=seed)))
