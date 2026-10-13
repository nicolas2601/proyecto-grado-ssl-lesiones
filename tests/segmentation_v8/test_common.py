import unittest,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
from dermssl.segmentation_methods.common.data import read_manifest,digest,make_dataset
from dermssl.segmentation_methods.common.training import gate_report,fit
from dermssl.segmentation_methods.common.losses import masked_bce
from dermssl.segmentation_methods.common.checkpoints import save_checkpoint,load_reviewed


class CommonTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.rows=[]
        for i in range(2):
            image=self.root/f'image{i}.png';mask=np.zeros((8,8),np.float32);mask[1:5,2:6]=1
            rgb=np.repeat((mask*255).astype(np.uint8)[...,None],3,axis=2);rgb[:,:,1]=i
            Image.fromarray(rgb).save(image);pseudo=self.root/f'pseudo{i}.npy';np.save(pseudo,mask)
            self.rows.append({'image_id':str(i),'dataset':'synthetic','image_path':str(image),'image_sha256':digest(image),
                              'pseudo_paths':[str(pseudo)],'pseudo_sha256s':[digest(pseudo)],
                              'pseudo_provenance':[{'method':'synthetic','producer_revision':'a'*40,'uses_expert_masks':False,'uses_diagnoses':False}]})
        self.doc={'schema_version':1,'partition':'train','count':2,'eligibility':{'verified':True,'excludes_target_test':True,
                 'patient_lesion_groups_preserved':True,'deduplication_preserved':True,'source_manifest_sha256':'a'*64,
                 'exclusions_sha256':'b'*64,'verification_revision':'c'*40},'records':self.rows}
        self.path=self.root/'manifest.json';self.write()
    def tearDown(self):self.temp.cleanup()
    def write(self):self.path.write_text(json.dumps(self.doc))
    def test_original_and_pseudo_hashes_and_aligned_geometry(self):
        document=read_manifest(self.path,need_pseudos=True,training=True)
        dataset=make_dataset(document,size=8,paired_flip=True)
        for epoch in range(3):
            dataset.set_epoch(epoch);item=dataset[0]
            torch.testing.assert_close(item['image'][0],item['target'][0])
    def test_resnet50_parameter_count_and_stage_geometry(self):
        from dermssl.segmentation_methods.common.networks import ResNet50Stages
        model=ResNet50Stages(last_stride=1).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()),23508032)
        with torch.no_grad():stages=model(torch.rand(1,3,64,64))
        self.assertEqual(tuple(t.shape[1] for t in stages),(256,512,1024,2048))
        self.assertEqual(tuple(t.shape[-1] for t in stages),(16,8,4,4))
    def test_local_review_overlay_checks_geometry_and_stays_local(self):
        from dermssl.segmentation_methods.common.review import save_review
        directory=self.root/'review';directory.mkdir()
        save_review(np.zeros((3,8,8)),np.ones((1,8,8)),np.ones((8,8)),directory)
        self.assertTrue((directory/'overlay_0.png').is_file())
        self.assertFalse(json.loads((directory/'legend.json').read_text())['upload_to_wandb'])
        with self.assertRaises(ValueError):save_review(np.zeros((3,8,8)),np.ones((1,7,7)),np.ones((8,8)),directory)
    def test_mutated_original_is_rejected(self):
        Path(self.rows[0]['image_path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):read_manifest(self.path)
    def test_test_partition_and_diagnoses_rejected(self):
        self.doc['partition']='target_test';self.write()
        with self.assertRaises(ValueError):read_manifest(self.path)
        self.doc['partition']='train';self.rows[0]['diagnosis']='forbidden';self.write()
        with self.assertRaises(ValueError):read_manifest(self.path)
    def test_expert_pseudo_and_empty_validity_rejected(self):
        self.rows[0]['pseudo_provenance'][0]['uses_expert_masks']=True;self.write()
        with self.assertRaises(ValueError):read_manifest(self.path,need_pseudos=True)
        with self.assertRaises(ValueError):masked_bce(torch.zeros(1,1,2,2),torch.ones(1,1,2,2),torch.zeros(1,1,2,2))
    def test_no_fixed_100_image_limit(self):
        records=[]
        for i in range(121):
            path=self.root/f'pool{i}.png';Image.fromarray(np.full((2,2,3),i,np.uint8)).save(path)
            records.append({'image_id':str(i),'dataset':'synthetic','image_path':str(path),'image_sha256':digest(path)})
        self.doc['records']=records;self.doc['count']=121;self.write()
        self.assertEqual(read_manifest(self.path)['count'],121)
    def test_source_normalization_is_identical_for_shallow_and_deep(self):
        from dermssl.segmentation_methods.common.bank import NormalizedEncoder
        class Identity(nn.Module):
            def forward(self,x):return x
            def shallow_features(self,x):return x
        model=NormalizedEncoder(Identity());x=torch.rand(1,3,8,8)
        torch.testing.assert_close(model(x),model.shallow_features(x))
        torch.testing.assert_close(model(x),(x-model.mean)/model.std)
    def test_cuda_cannot_start_without_a_measured_budget(self):
        from dermssl.segmentation_methods.common.training import prepare_device
        with patch('torch.cuda.is_available',return_value=True):
            with self.assertRaises(ValueError):prepare_device({'device':'cuda','resources':{}})
    def test_training_gate_cannot_be_satisfied_by_reading_a_paper(self):
        gate=self.root/'gate.json';gate.write_text(json.dumps({'method':'usl_net','input_manifest_sha256':digest(self.path),'verification_revision':'a'*40,'data_verified':True}))
        with self.assertRaises(ValueError):gate_report(gate,'usl_net',self.path)
    def test_weights_only_checkpoint_roundtrip_and_provenance(self):
        model=nn.Linear(2,1);path=self.root/'weights.pt'
        save_checkpoint(model,path,method='synthetic',revision='a'*40,manifest_sha=digest(self.path))
        restored=nn.Linear(2,1);load_reviewed(restored,path,path.with_suffix('.pt.json'),expected_method='synthetic')
        torch.testing.assert_close(model.weight,restored.weight)
        with self.assertRaises(ValueError):save_checkpoint(model,path,method='synthetic',revision='a'*40,manifest_sha=digest(self.path))
    def test_one_synthetic_epoch_exercises_manifest_loader_loss_and_checkpoint(self):
        class Mini(nn.Module):
            def __init__(self):super().__init__();self.conv=nn.Conv2d(3,1,1)
            def forward(self,x):return {'logits':self.conv(x)}
            @staticmethod
            def loss(output,target,valid):return masked_bce(output['logits'],target,valid)
        gate=self.root/'gate.json';gate.write_text(json.dumps({'method':'usl_net','input_manifest_sha256':digest(self.path),
            'verification_revision':'a'*40,**{key:True for key in ('data_verified','independent_anatomical_evaluation_resolved','resource_budget_measured','weights_provenance_verified','smoke_passed')}}))
        with patch('dermssl.segmentation_methods.common.training.model_for',return_value=Mini()):
            result=fit('usl_net',self.path,self.root/'run',{'epochs':1,'batch_size':2,'resolution':8,'device':'cpu'},gate,revision='a'*40)
        self.assertEqual(result[0]['sample_count'],2);self.assertTrue((self.root/'run/final.pt').is_file())


if __name__=='__main__':unittest.main()
