import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'scripts/audit_modular_followup.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
CFG={'low':.35,'high':.65,'min_area_fraction':.003,'max_center_distance':.5,'max_area_fraction':.6,'region_mode':'central','artifact_suppression':False}
def inputs():
 raw=np.zeros((32,32),np.float32);raw[10:22,10:22]=1
 return raw,np.ones(raw.shape,bool),np.full((32,32,3),128,np.uint8)
def test_weak_abstains_without_minmax_amplification():
 raw,f,rgb=inputs();raw=.5+raw*.005
 result=a.pilot(raw,f,rgb,CFG,'S1')
 assert not result.valid and result.reason=='weak_span_le_0.01'
 assert (result.mask==255).all()
def test_strong_s1_equals_s0():
 raw,f,rgb=inputs()
 assert np.array_equal(a.pilot(raw,f,rgb,CFG,'S1').mask,a.pilot(raw,f,rgb,CFG,'S0').mask)
def test_s2_only_disables_orientation():
 raw,f,rgb=inputs();raw=1-raw
 expected=a.um.saliency_to_pseudo_mask(a.sp.normalize_saliency(raw),normalize=False,field=f,artifact=None,**{k:v for k,v in CFG.items() if k!='artifact_suppression'})
 assert np.array_equal(a.pilot(raw,f,rgb,CFG,'S2').mask,expected.mask)
 assert not np.array_equal(a.pilot(raw,f,rgb,CFG,'S0').mask,expected.mask)
def test_s3_hair_always_ignored(monkeypatch):
 raw,f,rgb=inputs();hair=np.zeros(raw.shape,np.uint8);hair[:,15]=1
 monkeypatch.setattr(a.hr,'elongated_hair_mask',lambda _:hair)
 result=a.pilot(raw,f,rgb,CFG,'S3')
 assert (result.mask[hair>0]==255).all()
def test_nonfinite_abstains():
 raw,f,rgb=inputs();raw[0,0]=np.nan
 result=a.pilot(raw,f,rgb,CFG,'S0');assert not result.valid and result.reason=='nonfinite'
