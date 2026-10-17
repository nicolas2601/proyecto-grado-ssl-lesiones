import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from dermssl.segmentation_methods.common.data import digest
from dermssl.segmentation_methods.usl_net.seeds import fit_ccam,saliency_stability,select_candidate_epoch


class TinyEncoder(nn.Module):
    """Last two stage channels (1024+2048=3072) x 3 encoders = 9216, matching USLSeedGenerator's default CCAM input width."""
    channels=(4,8,1024,2048)
    def __init__(self,in_channels=3):
        super().__init__();self.layers=nn.ModuleList([nn.Conv2d(in_channels,4,3,padding=1),nn.Conv2d(4,8,3,stride=2,padding=1),nn.Conv2d(8,1024,3,stride=2,padding=1),nn.Conv2d(1024,2048,1)])
    def forward(self,x):
        result=[]
        for layer in self.layers:x=layer(x).relu();result.append(x)
        return tuple(result)


class SaliencyStabilityTest(unittest.TestCase):
    def test_identical_maps_are_perfectly_stable(self):
        probe=torch.rand(40)
        self.assertAlmostEqual(saliency_stability(probe,probe.clone()),1.,places=5)
    def test_unrelated_maps_are_less_stable_than_identical(self):
        a=torch.rand(200);b=torch.rand(200)
        self.assertLess(saliency_stability(a,b),saliency_stability(a,a.clone()))
    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):saliency_stability(torch.rand(4),torch.rand(5))
    def test_non_flat_rejected(self):
        with self.assertRaises(ValueError):saliency_stability(torch.rand(2,2),torch.rand(2,2))
    def test_constant_map_has_no_defined_correlation(self):
        with self.assertRaises(ValueError):saliency_stability(torch.ones(10),torch.rand(10))


class SelectCandidateEpochTest(unittest.TestCase):
    def test_first_epoch_crossing_threshold_wins(self):
        self.assertEqual(select_candidate_epoch([None,.5,.97,.99,.3]),4)
    def test_no_candidate_when_never_reached(self):
        self.assertIsNone(select_candidate_epoch([None,.1,.2]))
    def test_empty_history_has_no_candidate(self):
        self.assertIsNone(select_candidate_epoch([]))


class FitCcamResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.rows=[]
        for i in range(4):
            image=self.root/f'image{i}.png'
            rgb=np.full((8,8,3),i*20,np.uint8)
            Image.fromarray(rgb).save(image)
            self.rows.append({'image_id':str(i),'dataset':'synthetic','image_path':str(image),'image_sha256':digest(image)})
        self.doc={'schema_version':1,'partition':'train','count':4,'eligibility':{'verified':True,'excludes_target_test':True,
                 'patient_lesion_groups_preserved':True,'deduplication_preserved':True,'source_manifest_sha256':'a'*64,
                 'exclusions_sha256':'b'*64,'verification_revision':'c'*40},'records':self.rows}
        self.manifest=self.root/'manifest.json';self.manifest.write_text(json.dumps(self.doc))
        self.gate=self.root/'gate.json'
        self.gate.write_text(json.dumps({'method':'usl_ccam','input_manifest_sha256':digest(self.manifest),
            'verification_revision':'a'*40,**{key:True for key in
            ('data_verified','independent_anatomical_evaluation_resolved','resource_budget_measured',
             'weights_provenance_verified','smoke_passed')}}))
        self.config={'epochs':3,'batch_size':2,'resolution':8,'device':'cpu','seed':42}
        self.encoders=lambda:{name:TinyEncoder() for name in ('simclr','mocov1','mocov2')}
    def tearDown(self):self.temp.cleanup()
    def _fit(self,**kwargs):
        with patch('dermssl.segmentation_methods.usl_net.seeds.source_bank',return_value=self.encoders()):
            return fit_ccam(self.manifest,'unused-bank-path',self.root/'run',self.config,self.gate,revision='a'*40,**kwargs)
    def test_full_run_writes_final_checkpoint_and_per_epoch_stability(self):
        self._fit()
        self.assertTrue((self.root/'run/ccam.pt').is_file())
        rows=[json.loads(line) for line in (self.root/'run/metrics.jsonl').read_text().splitlines()]
        self.assertEqual([row['epoch'] for row in rows],[1,2,3])
        self.assertIsNone(rows[0]['saliency_stability'])
        self.assertIsInstance(rows[1]['saliency_stability'],float)
        self.assertIsInstance(rows[2]['saliency_stability'],float)
    def test_resume_continues_from_the_saved_epoch_without_duplication(self):
        # Simulate a crash right after epoch 1's resume.pt is written, before epoch 2 starts.
        from dermssl.segmentation_methods.usl_net import seeds as seeds_module
        original=seeds_module.atomic_checkpoint
        def crash_after_first_epoch(*args,**kwargs):
            original(*args,**kwargs);raise RuntimeError('simulated interruption')
        with patch('dermssl.segmentation_methods.usl_net.seeds.source_bank',return_value=self.encoders()):
            with patch('dermssl.segmentation_methods.usl_net.seeds.atomic_checkpoint',side_effect=crash_after_first_epoch):
                with self.assertRaises(RuntimeError):
                    fit_ccam(self.manifest,'unused-bank-path',self.root/'run',self.config,self.gate,revision='a'*40)
        self.assertTrue((self.root/'run/resume.pt').is_file())
        self.assertFalse((self.root/'run/ccam.pt').is_file())
        rows_after_crash=[json.loads(line) for line in (self.root/'run/metrics.jsonl').read_text().splitlines()]
        self.assertEqual([row['epoch'] for row in rows_after_crash],[1])
        with patch('dermssl.segmentation_methods.usl_net.seeds.source_bank',return_value=self.encoders()):
            fit_ccam(self.manifest,'unused-bank-path',self.root/'run',self.config,self.gate,revision='a'*40,resume=True)
        rows=[json.loads(line) for line in (self.root/'run/metrics.jsonl').read_text().splitlines()]
        self.assertEqual([row['epoch'] for row in rows],[1,2,3])
        self.assertTrue((self.root/'run/ccam.pt').is_file())
    def test_resume_without_existing_directory_rejected(self):
        with self.assertRaises(ValueError):self._fit(resume=True)
    def test_resume_identity_mismatch_rejected(self):
        partial_config=dict(self.config,epochs=1)
        with patch('dermssl.segmentation_methods.usl_net.seeds.source_bank',return_value=self.encoders()):
            fit_ccam(self.manifest,'unused-bank-path',self.root/'run',partial_config,self.gate,revision='a'*40)
        with patch('dermssl.segmentation_methods.usl_net.seeds.source_bank',return_value=self.encoders()):
            with self.assertRaises(ValueError):
                fit_ccam(self.manifest,'unused-bank-path',self.root/'run',self.config,self.gate,revision='b'*40,resume=True)


if __name__=='__main__':unittest.main()
