"""Controlled v8 entry point: check/synthetic/pretrain/ccam/generate/fit/predict."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,signal,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'plan-segmentacion/comun'))
from dermssl.segmentation_methods import METHODS


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--stage',choices=('check','synthetic','pretrain','ccam','generate','fit','predict','refine'),required=True)
    p.add_argument('--config',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--gate',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--predictions',type=Path,help='Completed predict output directory for refine');p.add_argument('--wandb-project',default='')
    p.add_argument('--resume',action='store_true',help='Resume a fit from its identity-checked checkpoint')
    p.add_argument('--ssl-origin',choices=('simclr','mocov1','mocov2','mocov3','dino'))
    p.add_argument('--include-common',action='store_true',help='Run shared contract tests once with a method smoke')
    p.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    return p.parse_args()


def configuration(args):
    path=args.config or ROOT/'src/dermssl/segmentation_methods'/args.method/'config.json'
    c=json.loads(path.read_text())
    if c.get('method')!=args.method or c.get('schema_version')!=1:raise ValueError('Method/config identity mismatch')
    if args.stage not in {'check','synthetic'} and c.get('execution_enabled') is not True:
        raise ValueError('Reviewed configuration required: execution_enabled remains false')
    return c


def storage_precheck(args,config):
    limit=config['resources'].get('max_output_mb')
    if not isinstance(limit,(int,float)) or limit<=0:raise ValueError('Explicit output storage budget required')
    if args.stage in {'generate','predict'}:
        document=json.loads(args.manifest.read_text());count=document.get('count')
        if type(count)is not int or count<=0:raise ValueError('Positive frozen manifest count required')
        sources=4 if args.method=='rpi_net' and args.stage=='generate' else 1
        pixels=config.get('resolution',224)**2
        # Conservative NPY + uncompressed local RGB/overlay estimate, no old files.
        estimate=count*(pixels*((sources+2)*4+(sources+1)*6)+16384)/(1024**2)
    elif args.stage=='refine':
        document=json.loads(args.manifest.read_text());count=document.get('count')
        if type(count)is not int or count<=0:raise ValueError('Positive frozen manifest count required')
        pixels=config.get('resolution',224)**2
        # One refined probability/pseudo map per row, plus a reserve for manifest metadata.
        estimate=count*(pixels*4+4096)/(1024**2)
    else:estimate=128 if args.stage=='synthetic' else 1024
    if estimate>limit:raise ValueError('Registered output budget too small for the entire manifest')
    parent=args.out.parent.resolve()
    while not parent.exists():parent=parent.parent
    free=shutil.disk_usage(parent).free/(1024**2)
    if free<estimate+512:raise ValueError('Insufficient free disk space for full output and reserve')
    return estimate


def budget_child(args,config):
    """One child process group; time/RSS limit affects only this run, not cluster jobs."""
    limits=config['resources'];seconds=limits.get('timeout_seconds');ram=limits.get('max_rss_mb')
    if not isinstance(seconds,(int,float)) or seconds<=0 or not isinstance(ram,(int,float)) or ram<=0:
        raise ValueError('Positive timeout and RSS budgets required')
    child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--_worker'],
                           cwd=ROOT,start_new_session=True)
    started=time.monotonic()
    try:
        while child.poll() is None:
            rss=0
            try:
                for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                    if line.startswith('VmRSS:'):rss=int(line.split()[1])/1024
            except FileNotFoundError:pass
            if time.monotonic()-started>seconds or rss>ram:
                os.killpg(child.pid,signal.SIGTERM)
                grace=limits.get('shutdown_grace_seconds',120)
                if not isinstance(grace,(int,float)) or grace<=0:raise ValueError('Positive shutdown grace budget required')
                try:child.wait(timeout=grace)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError('Registered wall-time/RSS budget exceeded; partial evidence preserved')
            time.sleep(.2)
        return child.returncode
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise


def main():
    args=arguments();config=configuration(args)
    if args.stage=='check':
        print(json.dumps({'method':args.method,'execution_enabled':config['execution_enabled'],
                          'stages':config['stages'],'resources':config['resources']},indent=2))
        if args.manifest:
            from dermssl.segmentation_methods.common.data import read_manifest
            document=read_manifest(args.manifest,need_pseudos=args.stage=='fit',sources=4 if args.method=='rpi_net' else 1)
            print(json.dumps({'manifest_count':document['count'],'partition':document['partition']}))
        return 0
    if not args.out:raise ValueError('Explicit output directory required')
    if args.resume and args.stage not in ('fit','ccam'):raise ValueError('--resume is currently supported only for fit and ccam')
    if args.resume:
        if not args.out.is_dir() or not (args.out/'resume.pt').is_file():raise ValueError('--resume requires an existing fit directory with resume.pt')
    elif args.out.exists():raise ValueError('New output directory required; existing outputs are never overwritten')
    if not args.wandb_project or os.environ.get('WANDB_MODE','').lower() in {'offline','disabled'}:
        raise ValueError('Online W&B project required for executed stages')
    if args.stage not in config['stages']:raise ValueError('Stage unsupported for this method')
    if args.stage=='refine' and args.method not in {'usl_net','rpi_net'}:
        raise ValueError('Refinement is only defined for USL-Net and RPI-Net')
    if args.stage=='refine' and not args.predictions:
        raise ValueError('--predictions must point to a completed predict output directory')
    if args.stage!='synthetic' and (not args.manifest or not args.gate):raise ValueError('Frozen manifest and gate evidence required')
    if args.stage!='synthetic':
        from dermssl.segmentation_methods.common.training import gate_report
        from dermssl.segmentation_methods.common.data import read_manifest
        # Validate before the budget child or tracking creates run artifacts.
        gate_method=('usl_ccam' if args.stage=='ccam' else
                     'ssl_'+args.ssl_origin if args.stage=='pretrain' and args.ssl_origin else args.method)
        gate_report(args.gate,gate_method,
                    args.manifest,training=args.stage in {'fit','ccam','pretrain'},
                    config=config,stage=args.stage)
        read_manifest(args.manifest,purpose=args.stage,
                      training=args.stage in {'fit','ccam','pretrain','refine'})
    storage_precheck(args,config)
    if not args._worker:return budget_child(args,config)
    from nucleo.wandb_logging import WandbSink,run_id
    expected_run_id=run_id('w3-four-methods-v8',args.out.name)
    if args.resume:
        import torch
        resume_meta=torch.load(args.out/'resume.pt',map_location='cpu',weights_only=True)
        if resume_meta.get('identity',{}).get('tracking_run_id')!=expected_run_id:
            raise ValueError('Resume output name does not match the checkpoint W&B run; refusing to create another run')
    log_dir=args.out.parent/(args.out.name+'-tracking')
    if args.resume:
        if not log_dir.is_dir():raise ValueError('--resume requires the existing tracking directory for the same W&B run')
    else:
        if log_dir.exists():raise ValueError('New tracking directory required')
        log_dir.mkdir(parents=True)
    os.environ['WANDB_DIR']=str(log_dir.resolve());os.environ['WANDB_DISABLE_CODE']='true';os.environ['WANDB_CONSOLE']='off'
    name=args.out.name
    source_hashes={}
    from hashlib import sha256
    for path in sorted((ROOT/'src/dermssl/segmentation_methods').rglob('*')):
        if path.is_file() and '__pycache__' not in str(path):source_hashes[str(path.relative_to(ROOT))]=sha256(path.read_bytes()).hexdigest()
    source_hashes['scripts/run_segmentation_v8.py']=sha256(Path(__file__).read_bytes()).hexdigest()
    for path in sorted((ROOT/'tests/segmentation_v8').glob('*.py')):source_hashes[str(path.relative_to(ROOT))]=sha256(path.read_bytes()).hexdigest()
    for dependency in ('src/dermssl/w3_components.py','src/dermssl/w3_usl_models.py','plan-segmentacion/camino-b-cnn/pseudo-mascaras/ccam_engine.py'):
        source_hashes[dependency]=sha256((ROOT/dependency).read_bytes()).hexdigest()
    from dermssl.segmentation_methods.common.protocol import protocol_metadata
    protocol_info=protocol_metadata(config) if args.stage!='synthetic' else {}
    sink=WandbSink(args.wandb_project,name,group='w3-four-methods-v8',tags=[args.method,args.stage],
                   config={'method':args.method,'stage':args.stage,'seed':config.get('seed',42),
                           'resolution':config.get('resolution',224),'scientific_efficacy_evaluation':False,
                           'source_hashes':source_hashes,**protocol_info})
    if not sink.enabled:raise RuntimeError('Tracking unavailable; no method execution started')
    started=time.monotonic();status='failed'
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if args.stage=='synthetic':
            import unittest
            suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/segmentation_v8'),pattern='test_'+args.method+'.py')
            if args.include_common:suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests/segmentation_v8'),pattern='test_common.py'))
            result=unittest.TextTestRunner(verbosity=2).run(suite)
            report={'method':args.method,'tests_run':result.testsRun,'tests_failed':len(result.failures),
                    'tests_errors':len(result.errors),'tests_skipped':len(result.skipped),'real_images_read':0,
                    'gpu_training_started':False,'code_revision':revision,'source_hashes':source_hashes,'wandb_url':sink._run.url}
            args.out.mkdir(parents=True);(args.out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
            sink.log({k:report[k] for k in ('tests_run','tests_failed','tests_errors','tests_skipped','real_images_read')})
            if not result.wasSuccessful() or result.testsRun==0:raise RuntimeError('Synthetic verification failed or no tests collected')
        elif args.stage=='fit':
            from dermssl.segmentation_methods.common import training as fit_module
            fit=fit_module.fit
            previous_term=signal.signal(signal.SIGTERM,fit_module.request_stop)
            previous_int=signal.signal(signal.SIGINT,fit_module.request_stop)
            try:fit(args.method,args.manifest,args.out,config,args.gate,sink=sink,revision=revision,resume=args.resume,tracking_run_id=getattr(sink._run,'id',None),source_hashes=source_hashes)
            finally:
                signal.signal(signal.SIGTERM,previous_term);signal.signal(signal.SIGINT,previous_int)
        elif args.stage=='refine':
            from dermssl.segmentation_methods.common.data import read_manifest,digest
            from dermssl.segmentation_methods.common.training import gate_report
            from dermssl.segmentation_methods.common.refinement import refined_manifest
            gate_report(args.gate,args.method,args.manifest,training=False,config=config,stage='refine')
            prediction_summary=json.loads((args.predictions/'summary.json').read_text())
            if prediction_summary.get('input_manifest_sha256')!=digest(args.manifest):
                raise ValueError('Prediction output belongs to a different input manifest')
            sources=4 if args.method=='rpi_net' else 1
            document=read_manifest(args.manifest,need_pseudos=True,sources=sources,training=True)
            refined=refined_manifest(document,args.predictions,args.out,args.method,revision,
                                     usl_policy=config.get('usl_policy'),manifest_path=args.manifest)
            report={'method':args.method,'stage':'refine','count':document['count'],
                    'input_manifest_sha256':digest(args.manifest),'output_manifest_sha256':digest(refined),
                    'producer_revision':revision,'prediction_manifest_sha256':prediction_summary['input_manifest_sha256']}
            (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            sink.log({'refined_count':report['count'],'output_manifest_sha256':report['output_manifest_sha256']})
        elif args.stage=='pretrain':
            if not args.ssl_origin:raise ValueError('Explicit SSL origin required')
            from dermssl.segmentation_methods.common.pretraining import local_pretrain
            local_pretrain(args.ssl_origin,args.manifest,args.out,config,args.gate,revision=revision,sink=sink)
        elif args.stage=='ccam':
            from dermssl.segmentation_methods.usl_net.seeds import fit_ccam
            fit_ccam(args.manifest,config['source_bank'],args.out,config,args.gate,revision=revision,sink=sink,
                     resume=args.resume,tracking_run_id=getattr(sink._run,'id',None),source_hashes=source_hashes)
        else:
            from dermssl.segmentation_methods.common.inference import export
            export(args.method,args.stage,args.manifest,args.out,config,args.gate,revision=revision,sink=sink)
        status='completed';return 0
    finally:
        sink.log({'completed':int(status=='completed'),'elapsed_seconds':time.monotonic()-started});sink.finish()


if __name__=='__main__':raise SystemExit(main())
