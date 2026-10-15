"""Contract fixtures for UNABIA only; no real images or model execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import hashlib

import numpy as np
from PIL import Image

from dermssl.segmentation_methods.common.data import digest, read_manifest
from dermssl.segmentation_methods.common.protocol import cohort_digest, validate_binding


class RetrospectiveFixture:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        image = self.root / 'image.png'
        Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image)
        self.rows = [{'image_id': 'ISIC_0000001', 'dataset': 'isic2018_task1',
                      'image_path': str(image), 'image_sha256': digest(image)}]
        self.protocol = {'schema_version': 1, 'protocol_id': 'fixture-r1',
                         'regime': 'retrospective_public_benchmark', 'review_revision': 'a'*40,
                         'accepted_by': 'fixture-reviewer', 'accepted_at': '2026-10-15',
                         'patient_independence_accredited': False,
                         'expert_masks_used_for_learning': False,
                         'diagnoses_used_for_learning': False,
                         'evaluation_cohort_sha256': cohort_digest(self.rows),
                         'reference_cohort_sha256': 'd'*64,
                         'reserved_images': [{k: self.rows[0][k] for k in ('image_id','image_sha256')}],
                         'learning_cohort_sha256s': ['e'*64],
                         'prediction_threshold': 0.5, 'threshold_inclusive': True,
                         'bootstrap_unit': 'image', 'bootstrap_seed': 42,
                         'bootstrap_replicates': 100, 'evidence': {}}
        for name in ('exclusions_verified', 'duplicates_adjudicated', 'mask_content_validated',
                     'exposure_review_complete', 'evaluation_masks_isolated'):
            path = self.root / (name + '.json')
            path.write_text(json.dumps({'check': name, 'verified': True,
                                        'review_revision': 'a'*40}))
            self.protocol['evidence'][name] = {'path': str(path), 'sha256': digest(path)}
        self.pin()
        self.doc = {'schema_version': 1, 'partition': 'evaluation', 'count': 1,
                    'evaluation_protocol': self.ref.copy(), 'records': self.rows,
                    'eligibility': {'verified': True, 'excludes_target_test': True,
                                    'patient_lesion_groups_preserved': False,
                                    'patient_independence_accredited': False,
                                    'deduplication_preserved': True,
                                    'source_manifest_sha256': 'b'*64,
                                    'exclusions_sha256': 'c'*64,
                                    'verification_revision': 'a'*40}}
        self.manifest = self.root / 'manifest.json'
        self.write_manifest()

    def pin(self):
        for reference in self.protocol['evidence'].values():
            receipt=json.loads(Path(reference['path']).read_text())
            for key in ('evaluation_cohort_sha256','reference_cohort_sha256','learning_cohort_sha256s'):
                receipt[key]=self.protocol[key]
            Path(reference['path']).write_text(json.dumps(receipt))
            reference['sha256']=digest(reference['path'])
        path = self.root / 'protocol.json'
        path.write_text(json.dumps(self.protocol))
        self.ref = {'path': str(path), 'sha256': digest(path), 'protocol_id': 'fixture-r1'}

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.doc))


class RetrospectiveProtocol(RetrospectiveFixture, unittest.TestCase):
    def test_reserved_images_are_available_only_to_prediction(self):
        self.assertEqual(read_manifest(self.manifest, purpose='predict')['count'], 1)
        for purpose in ('fit', 'generate', 'ccam', 'refine', 'pretrain'):
            with self.subTest(purpose=purpose), self.assertRaises(ValueError):
                read_manifest(self.manifest, purpose=purpose)

    def test_patient_claim_and_missing_adjudication_cannot_relax_gate(self):
        self.protocol['patient_independence_accredited'] = True
        self.pin()
        with self.assertRaises(ValueError):
            validate_binding({'evaluation_protocol': self.ref})
        self.protocol['patient_independence_accredited'] = False
        del self.protocol['evidence']['duplicates_adjudicated']
        self.pin()
        with self.assertRaises(ValueError):
            validate_binding({'evaluation_protocol': self.ref})

    def test_binding_rejects_different_config_and_tampered_evidence(self):
        other = dict(self.ref, protocol_id='different')
        with self.assertRaises(ValueError):
            validate_binding({'evaluation_protocol': self.ref}, {'evaluation_protocol': other})
        evidence = self.protocol['evidence']['exclusions_verified']['path']
        Path(evidence).write_text('{}')
        with self.assertRaises(ValueError):
            validate_binding({'evaluation_protocol': self.ref})

    def test_evaluation_rejects_pseudos_and_changed_cohort(self):
        self.rows[0]['pseudo_paths'] = []
        self.write_manifest()
        with self.assertRaises(ValueError):
            read_manifest(self.manifest, purpose='predict')
        del self.rows[0]['pseudo_paths']
        self.rows[0]['image_id'] = 'ISIC_0000002'
        self.write_manifest()
        with self.assertRaises(ValueError):
            read_manifest(self.manifest, purpose='predict')

    def test_training_cannot_relabel_reserved_images(self):
        self.doc['partition'] = 'train'
        self.write_manifest()
        with self.assertRaises(ValueError):
            read_manifest(self.manifest, purpose='fit', training=True)

    def test_refinement_direct_api_cannot_consume_evaluation(self):
        from dermssl.segmentation_methods.common.refinement import refined_manifest
        with self.assertRaises(ValueError):
            refined_manifest(self.doc, self.root/'missing', self.root/'refined', 'usl_net', 'a'*40)


class FrozenExportEvaluation(RetrospectiveFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        script = Path(__file__).resolve().parents[2] / 'scripts/evaluate_segmentation_v8.py'
        spec = importlib.util.spec_from_file_location('v8_eval_contract', script)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_threshold_tie_is_foreground_and_empty_metric_is_null(self):
        result = self.module.score_probability(np.array([[0.5, 0.49]]), np.array([[1, 0]]))
        self.assertEqual(result['dice'], 1.0)
        self.assertEqual(result['iou'], 1.0)
        empty = self.module.score_probability(np.zeros((1, 2)), np.zeros((1, 2)))
        self.assertIsNone(empty['sensitivity'])
        json.dumps(empty, allow_nan=False)

    def test_id_matching_rejects_duplicates_extras_and_missing(self):
        for rows in ([{'image_id': 'a'}, {'image_id': 'a'}], [{'image_id': 'b'}], []):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.module.exact_index(rows, {'a'})

    def test_failed_cases_keep_original_denominator(self):
        result = self.module.summarize_cases([
            {'image_id': 'a', 'status': 'ok', 'dice': 1., 'iou': 1., 'sensitivity': None},
            {'image_id': 'b', 'status': 'abstained', 'dice': 0., 'iou': 0., 'sensitivity': None},
        ], replicates=100, seed=42)
        self.assertEqual(result['denominator'], 2)
        self.assertEqual(result['failed_count'], 1)
        self.assertEqual(result['metrics']['dice']['mean'], .5)
        self.assertEqual(result['metrics']['sensitivity']['undefined_count'], 2)
        self.assertIsNone(result['metrics']['sensitivity']['mean'])
        self.assertEqual(result['bootstrap_unit'], 'image')
        json.dumps(result, allow_nan=False)

    def test_complete_export_scores_and_tampered_probability_aborts(self):
        from dermssl.segmentation_methods.common.protocol import object_digest
        mask=self.root/'mask.png'
        Image.fromarray(np.full((2,2),255,np.uint8)).save(mask)
        self.protocol['reference_cohort_sha256']=object_digest([
            ('ISIC_0000001',self.rows[0]['image_sha256'],digest(mask))])
        self.pin();self.doc['evaluation_protocol']=self.ref.copy();self.write_manifest()
        refs={'schema_version':1,'partition':'evaluation','evaluation_protocol':self.ref.copy(),
              'evaluation_cohort_sha256':self.protocol['evaluation_cohort_sha256'],'count':1,
              'records':[dict(image_id='ISIC_0000001',image_sha256=self.rows[0]['image_sha256'],
                              mask_path=str(mask),mask_sha256=digest(mask))]}
        reference=self.root/'references.json';reference.write_text(json.dumps(refs))
        export=self.root/'export';export.mkdir()
        config={'evaluation_protocol':self.ref.copy(),'resolution':2}
        (export/'config.json').write_text(json.dumps(config))
        gate={'method':'sled','input_manifest_sha256':digest(self.manifest),
              'evaluation_protocol':self.ref.copy(),'config_sha256':object_digest(config),
              'patient_independence_accredited':False,'retrospective_protocol_reviewed':True,
              'verification_revision':'a'*40,**{k:True for k in (
                  'data_verified','resource_budget_measured','weights_provenance_verified','smoke_passed')}}
        (export/'gate.json').write_text(json.dumps(gate))
        probability=export/'probability.npy';np.save(probability,np.full((2,2),.5))
        valid=export/'valid.npy';np.save(valid,np.ones((2,2)))
        case={'image_id':'ISIC_0000001','image_sha256':self.rows[0]['image_sha256'],
              'status':'ok','prediction_path':str(probability),'prediction_sha256':digest(probability),
              'valid_path':str(valid),'valid_sha256':digest(valid)}
        (export/'cases.jsonl').write_text(json.dumps(case)+'\n')
        summary={'stage':'predict','method':'sled','export_complete':True,'input_count':1,
                 'checkpoint_kind':'image_wise_no_checkpoint',
                 'successful_count':1,'abstentions':0,'input_manifest_sha256':digest(self.manifest),
                 'config_sha256':object_digest(config),'gate_sha256':digest(export/'gate.json'),
                 'cases_sha256':digest(export/'cases.jsonl'),'prediction_threshold':.5,
                 'threshold_inclusive':True,'evaluation_protocol':self.ref.copy()}
        (export/'summary.json').write_text(json.dumps(summary))
        report=self.module.evaluate_export(self.manifest,reference,export,code_revision='a'*40)
        self.assertEqual(report['metrics']['dice']['mean'],1.)
        self.assertFalse(report['patient_independence_accredited'])
        evaluator=Path(self.module.__file__)
        metrics=Path(self.module._metrics_module().__file__)
        self.assertEqual(report['evaluator_sha256'],hashlib.sha256(evaluator.read_bytes()).hexdigest())
        self.assertEqual(report['metrics_sha256'],hashlib.sha256(metrics.read_bytes()).hexdigest())
        self.assertEqual(report['code_revision'],'a'*40)
        np.save(probability,np.zeros((2,2)))
        with self.assertRaises(ValueError):
            self.module.evaluate_export(self.manifest,reference,export,code_revision='a'*40)

    def test_evaluator_requires_explicit_snapshot_revision(self):
        with self.assertRaises(ValueError):
            self.module.evaluate_export(self.manifest,self.root/'absent',self.root/'absent',code_revision=None)

    def test_neural_export_rejects_missing_or_substituted_lineage_receipt(self):
        import torch
        from dermssl.segmentation_methods.common.checkpoints import save_checkpoint
        path=self.root/'final.pt';model=torch.nn.Linear(1,1)
        lineage={'protocol_sha256':self.ref['sha256'],'training_manifest_sha256':'b'*64,
                 'training_gate_sha256':'c'*64,'training_cohort_sha256':'e'*64}
        save_checkpoint(model,path,method='usl_net',revision='a'*40,manifest_sha='b'*64,lineage=lineage)
        provenance=path.with_suffix('.pt.json')
        config={'evaluation_protocol':self.ref.copy(),'checkpoint':str(path),
                'checkpoint_provenance':str(provenance),'checkpoint_lineage':lineage,
                'checkpoint_sha256':digest(path),'checkpoint_provenance_sha256':digest(provenance)}
        with self.assertRaises(ValueError):
            self.module.verify_checkpoint_export(config,{},self.root,'usl_net')
        receipt={'method':'usl_net','checkpoint_sha256':digest(path),
                 'provenance_sha256':digest(provenance),'lineage':lineage,
                 'patient_independence_accredited':False}
        receipt_path=self.root/'checkpoint.lineage.json';receipt_path.write_text(json.dumps(receipt))
        summary={'checkpoint_kind':'trained','checkpoint_lineage_sha256':digest(receipt_path)}
        self.module.verify_checkpoint_export(config,summary,self.root,'usl_net')
        receipt['lineage']=dict(lineage,training_gate_sha256='f'*64)
        receipt_path.write_text(json.dumps(receipt))
        summary['checkpoint_lineage_sha256']=digest(receipt_path)
        with self.assertRaises(ValueError):
            self.module.verify_checkpoint_export(config,summary,self.root,'usl_net')

    def test_refinement_rejects_historical_export_and_inappropriate_checkpoint_kind(self):
        from dermssl.segmentation_methods.common.refinement import refined_manifest
        from dermssl.segmentation_methods.common.protocol import object_digest
        import torch
        from dermssl.segmentation_methods.common.checkpoints import save_checkpoint
        image=self.root/'learning.png';Image.fromarray(np.ones((2,2,3),np.uint8)).save(image)
        pseudo=self.root/'pseudo.npy';np.save(pseudo,np.zeros((2,2)))
        row={'image_id':'learning-1','dataset':'fixture','image_path':str(image),'image_sha256':digest(image),
             'pseudo_paths':[str(pseudo)],'pseudo_sha256s':[digest(pseudo)],
             'pseudo_provenance':[{'method':'usl_net','producer_revision':'a'*40,
                                   'uses_expert_masks':False,'uses_diagnoses':False}]}
        self.protocol['learning_cohort_sha256s']=[cohort_digest([row])];self.pin()
        self.doc.update(partition='train',records=[row],evaluation_protocol=self.ref.copy())
        self.write_manifest()
        weights=self.root/'trained.pt'
        lineage={'protocol_sha256':self.ref['sha256'],'training_manifest_sha256':digest(self.manifest),
                 'training_gate_sha256':'c'*64,'training_cohort_sha256':cohort_digest([row])}
        save_checkpoint(torch.nn.Linear(1,1),weights,method='usl_net',revision='a'*40,
                        manifest_sha=digest(self.manifest),lineage=lineage)
        sidecar=weights.with_suffix('.pt.json')
        config={'evaluation_protocol':self.ref.copy(),'checkpoint':str(weights),
                'checkpoint_provenance':str(sidecar),'checkpoint_lineage':lineage,
                'checkpoint_sha256':digest(weights),'checkpoint_provenance_sha256':digest(sidecar)}
        export=self.root/'prediction';export.mkdir();case_dir=export/'case_000000';case_dir.mkdir()
        probability=case_dir/'probability.npy';np.save(probability,np.zeros((2,2)))
        case={'image_id':'learning-1','image_sha256':digest(image),'status':'ok',
              'prediction_path':str(probability),'prediction_sha256':digest(probability)}
        (export/'cases.jsonl').write_text(json.dumps(case)+'\n')
        (export/'config.json').write_text(json.dumps(config))
        gate={'method':'usl_net','input_manifest_sha256':digest(self.manifest),
              'config_sha256':object_digest(config),'evaluation_protocol':self.ref.copy(),
              'patient_independence_accredited':False,'retrospective_protocol_reviewed':True,
              'verification_revision':'a'*40,**{k:True for k in (
                  'data_verified','resource_budget_measured','weights_provenance_verified','smoke_passed')}}
        (export/'gate.json').write_text(json.dumps(gate))
        receipt={'method':'usl_net','checkpoint_sha256':digest(weights),'provenance_sha256':digest(sidecar),
                 'lineage':lineage,'patient_independence_accredited':False}
        receipt_path=export/'checkpoint.lineage.json';receipt_path.write_text(json.dumps(receipt))
        summary={'method':'usl_net','stage':'predict','export_complete':True,'input_count':1,
                 'successful_count':1,'abstentions':0,'input_manifest_sha256':digest(self.manifest),
                 'input_cohort_sha256':cohort_digest([row]),'config_sha256':object_digest(config),
                 'gate_sha256':digest(export/'gate.json'),'cases_sha256':digest(export/'cases.jsonl'),
                 'evaluation_protocol':self.ref.copy(),'checkpoint_kind':'trained',
                 'checkpoint_lineage_sha256':digest(receipt_path)}
        from dermssl.segmentation_methods.common.checkpoints import verify_prediction_export
        (export/'summary.json').write_text(json.dumps(summary))
        accepted=verify_prediction_export(self.doc,self.manifest,export,'usl_net')
        self.assertEqual(accepted['input_manifest_sha256'],digest(self.manifest))
        for name in ('historical','wrong-kind','substituted-receipt'):
            changed=json.loads(json.dumps(summary))
            if name=='historical':del changed['evaluation_protocol']
            if name=='wrong-kind':changed['checkpoint_kind']='image_wise_no_checkpoint'
            if name=='substituted-receipt':
                receipt['lineage']=dict(lineage,training_gate_sha256='f'*64)
                receipt_path.write_text(json.dumps(receipt));changed['checkpoint_lineage_sha256']=digest(receipt_path)
            (export/'summary.json').write_text(json.dumps(changed))
            with self.subTest(name=name),self.assertRaises(ValueError):
                refined_manifest(self.doc,export,self.root/name,'usl_net','a'*40,manifest_path=self.manifest)


class CheckpointLineage(unittest.TestCase):
    def test_legacy_checkpoint_cannot_gain_lineage_by_substituting_sidecar(self):
        import torch
        from dermssl.segmentation_methods.common.checkpoints import save_checkpoint,load_reviewed
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.pt';model=torch.nn.Linear(1,1)
            save_checkpoint(model,path,method='usl_net',revision='a'*40,manifest_sha='b'*64)
            origin_path=path.with_suffix('.pt.json')
            origin=json.loads(origin_path.read_text())
            lineage={'protocol_sha256':'c'*64,'training_manifest_sha256':'b'*64,
                     'training_gate_sha256':'d'*64,'training_cohort_sha256':'e'*64}
            origin['lineage']=lineage;origin_path.write_text(json.dumps(origin))
            with self.assertRaises(ValueError):
                load_reviewed(model,path,origin_path,expected_method='usl_net',expected_lineage=lineage)

    def test_checkpoint_lineage_mismatch_aborts_before_loading_weights(self):
        import torch
        from dermssl.segmentation_methods.common.checkpoints import save_checkpoint,load_reviewed
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'new.pt';model=torch.nn.Linear(1,1)
            lineage={'protocol_sha256':'c'*64,'training_manifest_sha256':'b'*64,
                     'training_gate_sha256':'d'*64,'training_cohort_sha256':'e'*64}
            save_checkpoint(model,path,method='usl_net',revision='a'*40,manifest_sha='b'*64,lineage=lineage)
            changed=dict(lineage,training_gate_sha256='f'*64)
            with self.assertRaises(ValueError):
                load_reviewed(model,path,path.with_suffix('.pt.json'),expected_lineage=changed)
            loaded=load_reviewed(model,path,path.with_suffix('.pt.json'),expected_lineage=lineage)
            self.assertEqual(loaded['lineage']['training_gate_sha256'],'d'*64)
            self.assertFalse(loaded['patient_independence_accredited'])


if __name__ == '__main__':
    unittest.main()
