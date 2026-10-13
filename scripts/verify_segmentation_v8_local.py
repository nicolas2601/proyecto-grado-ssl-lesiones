"""Offline synthetic verification; explicitly never claims W&B or cluster execution."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,importlib.util,json,platform,sys,time,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise ValueError('New evidence file required')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/segmentation_v8'),pattern='test_*.py')
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
    versions={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','scikit-learn','scikit-image','pillow')}
    hashes={}
    paths=list((ROOT/'src/dermssl/segmentation_methods').rglob('*.py'))+list((ROOT/'tests/segmentation_v8').glob('*.py'))+[ROOT/'scripts/run_segmentation_v8.py',Path(__file__).resolve()]
    for p in sorted(paths):hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for module in ('w3_components','w3_usl_models'):
        path=Path(importlib.util.find_spec('dermssl.'+module).origin)
        hashes['src/dermssl/'+module+'.py']=hashlib.sha256(path.read_bytes()).hexdigest()
    path=ROOT/'plan-segmentacion/camino-b-cnn/pseudo-mascaras/ccam_engine.py'
    hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    report={'status':'synthetic_passed' if result.wasSuccessful() else 'synthetic_failed','tests_run':result.testsRun,
            'tests_failed':len(result.failures),'tests_errors':len(result.errors),'tests_skipped':len(result.skipped),
            'seconds':time.monotonic()-started,'python':platform.python_version(),'versions':versions,'source_hashes':hashes,
            'real_images_read':0,'expert_masks_read':False,'diagnoses_read':False,'scientific_training_started':False,
            'synthetic_optimizer_steps_tested':True,'gpu_used':False,'wandb_run_created':False,
            'cluster_verification':'pending; browser native credential recovery blocked',
            'not_anatomical_efficacy_evaluation':True}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'},indent=2))
    return 0 if result.wasSuccessful() and result.testsRun>0 else 1


if __name__=='__main__':raise SystemExit(main())
