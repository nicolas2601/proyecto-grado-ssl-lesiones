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
    p.add_argument('--stage',choices=('check','synthetic','pretrain','ccam','generate','fit','predict'),required=True)
    p.add_argument('--config',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--gate',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--wandb-project',default='')
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
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise RuntimeError('Registered wall-time/RSS budget exceeded; partial evidence preserved')
            time.sleep(.2)
        return child.returncode
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
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
    if not args.out or args.out.exists():raise ValueError('New explicit output directory required')
    if not args.wandb_project or os.environ.get('WANDB_MODE','').lower() in {'offline','disabled'}:
        raise ValueError('Online W&B project required for executed stages')
    if args.stage not in config['stages']:raise ValueError('Stage unsupported for this method')
    if args.stage!='synthetic' and (not args.manifest or not args.gate):raise ValueError('Frozen manifest and gate evidence required')
    storage_precheck(args,config)
    if not args._worker:return budget_child(args,config)
    from nucleo.wandb_logging import WandbSink
    log_dir=args.out.parent/(args.out.name+'-tracking')
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
    sink=WandbSink(args.wandb_project,name,group='w3-four-methods-v8',tags=[args.method,args.stage],
                   config={'method':args.method,'stage':args.stage,'seed':config.get('seed',42),
                           'resolution':config.get('resolution',224),'scientific_efficacy_evaluation':False,'source_hashes':source_hashes})
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
            from dermssl.segmentation_methods.common.training import fit
            fit(args.method,args.manifest,args.out,config,args.gate,sink=sink)
        elif args.stage=='pretrain':
            if not args.ssl_origin:raise ValueError('Explicit SSL origin required')
            from dermssl.segmentation_methods.common.pretraining import local_pretrain
            local_pretrain(args.ssl_origin,args.manifest,args.out,config,args.gate,revision=revision,sink=sink)
        elif args.stage=='ccam':
            from dermssl.segmentation_methods.usl_net.seeds import fit_ccam
            fit_ccam(args.manifest,config['source_bank'],args.out,config,args.gate,revision=revision,sink=sink)
        else:
            from dermssl.segmentation_methods.common.inference import export
            export(args.method,args.stage,args.manifest,args.out,config,args.gate,revision=revision,sink=sink)
        status='completed';return 0
    finally:
        sink.log({'completed':int(status=='completed'),'elapsed_seconds':time.monotonic()-started});sink.finish()


if __name__=='__main__':raise SystemExit(main())
