"""CCAM-head training and text pseudo-label export; no expert masks."""
from pathlib import Path
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from ..common.bank import source_bank
from ..common.data import read_manifest,rgb_tensor_array,digest
from ..common.training import gate_report,seed_all,prepare_device,atomic_checkpoint,rng_state,restore_rng,_json_hash
from ..common.checkpoints import training_lineage
from .model import ContrastiveFusion,USLSeedGenerator
from .ccam import existing_ccam_loss

# Anchored to the 30-epoch MoCo preentrenamiento precedent already used in
# this project (docs/55); not a measured CCAM-specific convergence bound.
RECOMMENDED_EPOCH_CEILING=30


def saliency_stability(previous,current):
    """Pearson correlation in [-1, 1] between two flat saliency probes; 1 means unchanged."""
    if previous.shape!=current.shape or previous.dim()!=1:raise ValueError('Matching flat saliency vectors required')
    if previous.numel()<2:raise ValueError('At least two saliency values required for correlation')
    centered_previous=previous-previous.mean();centered_current=current-current.mean()
    denominator=centered_previous.norm()*centered_current.norm()
    if denominator<=0:raise ValueError('Degenerate (constant) saliency map has no defined correlation')
    return float((centered_previous@centered_current/denominator).clamp(-1.,1.))


def select_candidate_epoch(stability_history,threshold=.98):
    """Earliest epoch (1-indexed) whose saliency stability reaches ``threshold``; None if never."""
    for index,value in enumerate(stability_history):
        if value is not None and value>=threshold:return index+1
    return None


def _atomic_json(path,value,*,jsonl=False):
    temporary=path.with_name(path.name+'.tmp')
    text=(''.join(json.dumps(row)+'\n' for row in value) if jsonl
          else json.dumps(value,indent=2)+'\n')
    temporary.write_text(text,encoding='utf-8');os.replace(temporary,path)


def _source_bank_identity(bank_path):
    document=json.loads(Path(bank_path).read_text())
    # source_bank loads these exact paths relative to cwd, not the bank directory.
    assets={origin:{kind:{'path':str(Path(source[kind]).resolve()),
                          'sha256':digest(source[kind])}
                    for kind in ('checkpoint','provenance')}
            for origin,source in document['sources'].items()}
    return {'manifest_sha256':digest(bank_path),'assets':assets}


def _export_head(path,state,*,identity,epoch,lineage,status,adaptation_profile='legacy_v8'):
    """Recoverable run-owned export; a different run must never overwrite it."""
    if path.exists():
        existing=torch.load(path,map_location='cpu',weights_only=True)
        if existing.get('extra',{}).get('identity')!=identity:
            raise ValueError('Refusing to overwrite a CCAM export from another identity')
    bank_sha=identity['source_bank_identity']['manifest_sha256']
    extra={'identity':identity,'epoch':epoch,'selection_status':status,
           'adaptation_profile':adaptation_profile,'source_bank_manifest_sha256':bank_sha,
           'fusion_rms_epsilon':identity.get('fusion_rms_epsilon',1e-6)}
    payload={'model':state,'extra':extra,'lineage':lineage}
    origin={'method':'usl_ccam','protocol':'w3-four-methods-v8',
            'producer_revision':identity['revision'],'input_manifest_sha256':identity['manifest_sha256'],
            'uses_expert_masks':False,'uses_diagnoses':False,'input_normalization':'unit_rgb',
            'epoch':epoch,'selection_status':status,'source_hashes':identity['source_hashes'],
            'adaptation_profile':adaptation_profile,'source_bank_manifest_sha256':bank_sha,
            'fusion_rms_epsilon':identity.get('fusion_rms_epsilon',1e-6)}
    if lineage is not None:
        payload['patient_independence_accredited']=False
        origin['lineage']=lineage;origin['patient_independence_accredited']=False
    atomic_checkpoint(path,payload);origin['checkpoint_sha256']=digest(path)
    _atomic_json(path.with_suffix(path.suffix+'.json'),origin)
    return origin


def fit_ccam(manifest_path,bank_path,out,config,gate_path,*,revision,sink=None,resume=False,tracking_run_id=None,source_hashes=None):
    if config.get('ccam_variant') == 'ccam_stabilized_v1':
        return _fit_ccam_stabilized(manifest_path,bank_path,out,config,gate_path,revision=revision,
                                    sink=sink,resume=resume,tracking_run_id=tracking_run_id,
                                    source_hashes=source_hashes)
    gate_report(gate_path,'usl_ccam',manifest_path,config=config,stage='ccam')
    document=read_manifest(manifest_path,training=True,purpose='ccam')
    out=Path(out);resume_path=out/'resume.pt'
    if resume:
        if not out.is_dir() or not resume_path.is_file():raise ValueError('--resume requires an existing CCAM directory and intact resume.pt')
    elif out.exists():raise ValueError('New CCAM output directory required')
    size=config.get('resolution',224);batch=config.get('batch_size');epochs=config.get('epochs')
    if type(batch)is not int or batch<2 or type(epochs)is not int or epochs<1:raise ValueError('Explicit CCAM budget and batch >=2 required')
    if len(document['records'])<2 or len(document['records'])%batch==1:raise ValueError('Avoid singleton final CCAM batch without dropping inputs')
    selection=config.get('ccam_selection',{'rule':'first_stability_threshold','threshold':.98})
    threshold=selection.get('threshold')
    if (selection.get('rule')!='first_stability_threshold' or isinstance(threshold,bool)
            or not isinstance(threshold,(int,float)) or not -1<=threshold<=1):
        raise ValueError('CCAM requires first_stability_threshold with a finite Pearson threshold in [-1,1]')
    if resume and not source_hashes:raise ValueError('CCAM resume requires explicit source_hashes; legacy unbound checkpoints are not resumable here')
    seed=config.get('seed',42);seed_all(seed);device=prepare_device(config)
    bank_identity=_source_bank_identity(bank_path)
    encoders=source_bank(bank_path,('simclr','mocov1','mocov2'))
    if _source_bank_identity(bank_path)!=bank_identity:
        raise ValueError('Source bank manifest or checkpoint/provenance assets changed during loading')
    model=USLSeedGenerator(fusion=ContrastiveFusion(encoders)).to(device)
    class Originals(Dataset):
        def __len__(self):return len(document['records'])
        def __getitem__(self,i):return torch.from_numpy(rgb_tensor_array(document['records'][i],size))
    loader_generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(Originals(),batch_size=batch,shuffle=True,num_workers=0,generator=loader_generator)
    optimizer=torch.optim.AdamW(model.ccam.parameters(),lr=config.get('learning_rate',1e-3),
                               weight_decay=config.get('weight_decay',.01))
    identity={'method':'usl_ccam','revision':revision,'config_sha256':_json_hash(config),
              'manifest_sha256':digest(manifest_path),'gate_sha256':digest(gate_path),'tracking_run_id':tracking_run_id,
              'source_hashes':dict(source_hashes or {}),'source_bank_identity':bank_identity}
    probe_count=min(8,len(document['records']))
    with torch.no_grad():
        probe_images=torch.stack([torch.from_numpy(rgb_tensor_array(document['records'][i],size))
                                  for i in range(probe_count)]).to(device)
    if resume:
        # RNG states must remain CPU ByteTensors; loaders move model/optimizer state.
        payload=torch.load(resume_path,map_location='cpu',weights_only=True)
        if payload.get('identity')!=identity:raise ValueError('Resume checkpoint identity mismatch: code, config, gate, data or revision changed')
        model.ccam.load_state_dict(payload['ccam']);optimizer.load_state_dict(payload['optimizer'])
        start_epoch=int(payload['epoch']);loss_history=list(payload.get('loss_history',[]))
        stability_history=list(payload.get('stability_history',[]))
        previous_saliency=payload.get('previous_saliency')
        if previous_saliency is not None:previous_saliency=previous_saliency.to(device)
        records=list(payload['epoch_records']);selected=payload.get('selected_candidate')
        restore_rng(payload['rng'],loader_generator)
    else:
        out.mkdir(parents=True);start_epoch=0;loss_history=[];stability_history=[];previous_saliency=None
        records=[];selected=None
    # The committed checkpoint is authoritative after any logging interruption.
    _atomic_json(out/'metrics.jsonl',records,jsonl=True)
    for epoch in range(start_epoch,epochs):
        model.train();values=[]
        for images in loader:
            images=images.to(device);features=model.fusion(images)
            background=torch.sigmoid(model.ccam.background(features));foreground=1-background
            loss=existing_ccam_loss(features,foreground,config.get('ccam_alpha',.25))
            if not torch.isfinite(loss):raise ValueError('Nonfinite CCAM objective')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.ccam.parameters()):raise ValueError('Nonfinite CCAM gradient')
            optimizer.step();values.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():current_saliency=model(probe_images).detach().flatten()
        stability=None if previous_saliency is None else saliency_stability(previous_saliency,current_saliency)
        previous_saliency=current_saliency
        mean_loss=float(np.mean(values));loss_history.append(mean_loss);stability_history.append(stability)
        record={'epoch':epoch+1,'mean_ccam_loss':mean_loss,'sample_count':len(document['records']),
                'saliency_stability':stability}
        records.append(record)
        candidate_epoch=select_candidate_epoch(stability_history,threshold)
        if selected is None and candidate_epoch==epoch+1:
            selected={'epoch':epoch+1,'ccam':{name:value.detach().cpu().clone()
                       for name,value in model.ccam.state_dict().items()}}
        atomic_checkpoint(resume_path,{'identity':identity,'ccam':model.ccam.state_dict(),'optimizer':optimizer.state_dict(),
            'epoch':epoch+1,'loss_history':loss_history,'stability_history':stability_history,
            'epoch_records':records,'selected_candidate':selected,
            'previous_saliency':previous_saliency,'rng':rng_state(loader_generator)})
        _atomic_json(out/'metrics.jsonl',records,jsonl=True)
        # Logical epoch is explicit. External tracking is not an exactly-once transaction.
        if sink is not None:sink.log(record)
    lineage=training_lineage(document,manifest_path,gate_path)
    final=_export_head(out/'ccam.pt',model.ccam.state_dict(),identity=identity,epoch=epochs,
                       lineage=lineage,status='budget_final_not_selected')
    selected_origin=None
    if selected is not None:
        selected_origin=_export_head(out/'selected.ccam.pt',selected['ccam'],identity=identity,
                                     epoch=selected['epoch'],lineage=lineage,status='stability_candidate')
    _atomic_json(out/'selection.json',{'rule':selection['rule'],'threshold':threshold,
        'budget_epochs':epochs,'stopping_reason':'fixed_budget_exhausted','early_stopping':False,
        'selected_epoch':selected['epoch'] if selected is not None else None,
        'status':'stability_candidate' if selected is not None else 'threshold_not_reached',
        'final_checkpoint':{'path':'ccam.pt','sha256':final['checkpoint_sha256']},
        'selected_checkpoint':None if selected_origin is None else {
            'path':'selected.ccam.pt','sha256':selected_origin['checkpoint_sha256']},
        'scientific_convergence_verified':False,'anatomical_quality_verified':False})
    return final


def _pearson(left, right):
    left=left.double().flatten();right=right.double().flatten()
    left=left-left.mean();right=right-right.mean()
    denominator=left.norm()*right.norm()
    if not torch.isfinite(denominator) or denominator==0:return None
    return float((left@right/denominator).clamp(-1,1))


def _stabilized_monitor(model,images,device,policy):
    from dermssl.w3_components import DegenerateCase,usl_text_pseudolabel
    with torch.no_grad():saliency=model(images.to(device)).float().cpu()[:,0]
    areas=[];eligible=[];exact_zero=[];exact_one=[];constant=[];background_nonzero=[]
    foreground_fraction=[];background_fraction=[];uncertain_fraction=[]
    for value in saliency:
        array=value.numpy()
        exact_zero.append(float((value==0).float().mean()))
        exact_one.append(float((value==1).float().mean()))
        foreground_fraction.append(float((value>policy.high).float().mean()))
        background_fraction.append(float((value<policy.low).float().mean()))
        uncertain_fraction.append(float(((value>=policy.low)&(value<=policy.high)).float().mean()))
        constant.append(bool(torch.equal(value,value[:1,:1].expand_as(value))))
        background_nonzero.append(bool((value<policy.low).any()))
        try:
            pseudo,_=usl_text_pseudolabel(array,policy)
            foreground=pseudo==1
            areas.append(int(foreground.sum()));eligible.append(bool(foreground.sum()>=policy.min_area))
        except DegenerateCase:
            areas.append(0);eligible.append(False)
    return {'saliency':saliency,'exact_zero_fraction':exact_zero,'exact_one_fraction':exact_one,
            'constant_map':constant,'foreground_component_area':areas,'eligible_foreground':eligible,
            'background_nonzero':background_nonzero,'foreground_fraction':foreground_fraction,
            'background_fraction':background_fraction,'uncertain_fraction':uncertain_fraction}


def _fit_ccam_stabilized(manifest_path,bank_path,out,config,gate_path,*,revision,sink=None,
                         resume=False,tracking_run_id=None,source_hashes=None):
    """Isolated stabilized head profile; legacy CCAM fitting remains unchanged."""
    import resource
    import time
    from dermssl.w3_components import USLTextPolicy
    from .ccam_loss import ccam_loss_components
    gate_report(gate_path,'usl_ccam',manifest_path,config=config,stage='ccam')
    if (config.get('fusion_profile') != {'kind':'per_image_block_rms',
            'blocks':'simclr_stage3,simclr_stage4,mocov1_stage3,mocov1_stage4,mocov2_stage3,mocov2_stage4',
            'axes':[1,2,3],'epsilon':1e-6}
            or config.get('head_profile') != {'kind':'per_image_spatial_logit_standardization',
            'axes':[2,3],'variance_unbiased':False,'epsilon':1e-6,'affine':False,
            'running_statistics':False,'head':'existing_1x1_convolution'}
            or config.get('effective_batch_size') != 64
            or config.get('encoder_micro_batch_size') != 2
            or config.get('fusion_rms_epsilon') != 1e-6):
        raise ValueError('Stabilized CCAM profile, effective batch, or encoder micro-batch differs from the reviewed design')
    manifest_pin=config.get('training_manifest',{})
    if (Path(manifest_pin.get('path','')).resolve()!=Path(manifest_path).resolve()
            or manifest_pin.get('sha256')!=digest(manifest_path)):
        raise ValueError('Stabilized CCAM learning manifest differs from its configured pin')
    if (Path(config.get('source_bank','')).resolve()!=Path(bank_path).resolve()
            or config.get('source_bank_sha256')!=digest(bank_path)):
        raise ValueError('Stabilized CCAM source bank differs from its configured pin')
    protocol_pin=config.get('evaluation_protocol',{})
    if digest(protocol_pin.get('path',''))!=protocol_pin.get('sha256'):
        raise ValueError('Stabilized CCAM evaluation protocol differs from its configured pin')
    document=read_manifest(manifest_path,training=True,purpose='ccam')
    if len(document['records']) != manifest_pin.get('record_count'):
        raise ValueError('Stabilized CCAM learning record count differs from its configured pin')
    out=Path(out);resume_path=out/'resume.pt'
    if resume:
        if not out.is_dir() or not resume_path.is_file():raise ValueError('--resume requires an existing stabilized CCAM resume.pt')
    elif out.exists():raise ValueError('New stabilized CCAM output directory required')
    size=config.get('resolution',224);effective=config.get('effective_batch_size');micro=config.get('encoder_micro_batch_size')
    epochs=config.get('epochs');
    if (type(effective)is not int or effective<2 or type(micro)is not int or micro<1
            or type(epochs)is not int or epochs<1):raise ValueError('Explicit effective/micro batch and epoch budget required')
    if len(document['records'])%effective==1:raise ValueError('Avoid singleton final effective CCAM batch without dropping inputs')
    selection=config.get('ccam_selection',{})
    if (selection.get('rule')!='three_consecutive_eligible_stable_epochs'
            or selection.get('global_pearson_threshold')!=.98
            or selection.get('required_consecutive_epochs')!=3):
        raise ValueError('Stabilized CCAM selection rule is not the predeclared rule')
    seed=config.get('seed',42);seed_all(seed);device=prepare_device(config)
    profile='ccam_stabilized_v1';epsilon=config.get('fusion_rms_epsilon',1e-6)
    bank_identity=_source_bank_identity(bank_path);encoders=source_bank(bank_path,('simclr','mocov1','mocov2'))
    if _source_bank_identity(bank_path)!=bank_identity:raise ValueError('Source bank changed during loading')
    model=USLSeedGenerator(fusion=ContrastiveFusion(encoders,profile=profile,rms_epsilon=epsilon),profile=profile).to(device)
    class Originals(Dataset):
        def __len__(self):return len(document['records'])
        def __getitem__(self,i):return torch.from_numpy(rgb_tensor_array(document['records'][i],size))
    loader_generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(Originals(),batch_size=effective,shuffle=True,num_workers=0,generator=loader_generator)
    optimizer=torch.optim.AdamW(model.ccam.parameters(),lr=config.get('learning_rate',1e-3),
                               weight_decay=config.get('weight_decay',.01))
    identity={'method':'usl_ccam','adaptation_profile':profile,'fusion_rms_epsilon':epsilon,'revision':revision,
              'config_sha256':_json_hash(config),'manifest_sha256':digest(manifest_path),
              'gate_sha256':digest(gate_path),'tracking_run_id':tracking_run_id,
              'source_hashes':dict(source_hashes or {}),'source_bank_identity':bank_identity}
    probe_count=min(8,len(document['records']))
    if probe_count < 8:raise ValueError('Stabilized CCAM requires the fixed eight-image learning monitor')
    probe_images=torch.stack([torch.from_numpy(rgb_tensor_array(document['records'][i],size))
                              for i in range(probe_count)])
    policy=USLTextPolicy(**config.get('usl_policy',{}))
    if resume:
        payload=torch.load(resume_path,map_location='cpu',weights_only=True)
        if payload.get('identity')!=identity:raise ValueError('Stabilized resume identity mismatch')
        if (payload.get('terminal_guard') is not None
                or int(payload.get('collapse_streak',0)) >= config['collapse_guard']['required_consecutive_epochs']):
            raise ValueError('This stabilized CCAM checkpoint ended at a terminal collapse guard; start a new reviewed run')
        termination_path=out/'termination.json'
        if termination_path.exists():
            termination=json.loads(termination_path.read_text())
            if termination.get('status')=='stopped_non_success':
                raise ValueError('A terminal collapse receipt exists; this stabilized CCAM run cannot resume')
        model.ccam.load_state_dict(payload['ccam'],strict=True);optimizer.load_state_dict(payload['optimizer'])
        start_epoch=int(payload['epoch']);records=list(payload.get('epoch_records',[]))
        selected=payload.get('selected_candidate');previous=payload.get('previous_monitor')
        candidate_streak=int(payload.get('candidate_streak',0));collapse_streak=int(payload.get('collapse_streak',0))
        restore_rng(payload['rng'],loader_generator)
    else:
        out.mkdir(parents=True);start_epoch=0;records=[];selected=None;previous=None
        candidate_streak=0;collapse_streak=0
    _atomic_json(out/'metrics.jsonl',records,jsonl=True)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch,epochs):
        epoch_started=time.monotonic()
        model.train();epoch_loss=[];components={k:[] for k in ('positive_foreground','positive_background','negative')}
        grad_norms=[];gradient_nonzero_counts=[];epoch_samples=0
        for images in loader:
            images=images.to(device);feature_parts=[]
            with torch.no_grad():
                for start in range(0,len(images),micro):
                    feature_parts.append(model.fusion(images[start:start+micro]))
            features=torch.cat(feature_parts,dim=0)
            logits=model.ccam.logits(features)
            foreground=1-torch.sigmoid(logits)
            loss_parts=ccam_loss_components(features,foreground,config.get('ccam_alpha',.25))
            loss=loss_parts['total']
            if not torch.isfinite(loss):raise ValueError('Nonfinite stabilized CCAM objective')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gradients=[p.grad.detach().double() for p in model.ccam.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise ValueError('Missing or nonfinite stabilized CCAM gradients')
            grad_norm=torch.sqrt(sum(g.square().sum() for g in gradients))
            if not torch.isfinite(grad_norm):raise ValueError('Nonfinite stabilized CCAM gradient norm')
            grad_norms.append(float(grad_norm.cpu()));optimizer.step()
            gradient_nonzero_counts.append(sum(int(torch.count_nonzero(g).cpu()) for g in gradients))
            if epoch==0 and not (out/'first-update.json').exists():
                _atomic_json(out/'first-update.json',{'status':'first_optimizer_update_completed',
                    'epoch':1,'optimizer_update':1,'effective_batch_size':effective,
                    'sample_count':len(images),'adaptation_profile':profile,
                    'config_sha256':identity['config_sha256'],'manifest_sha256':identity['manifest_sha256']})
            epoch_loss.append(float(loss.detach().cpu()))
            for name in components:components[name].append(float(loss_parts[name].detach().cpu()))
            epoch_samples+=len(images)
        model.eval();current=_stabilized_monitor(model,probe_images,device,policy)
        per_image_corr=[]
        if previous is not None:
            per_image_corr=[_pearson(a,b) for a,b in zip(previous['saliency'],current['saliency'],strict=True)]
        global_corr=_pearson(previous['saliency'],current['saliency']) if previous is not None else None
        candidate_quality=(global_corr is not None and global_corr>=.98 and all(current['eligible_foreground'])
                           and all(current['background_nonzero']) and not all(current['constant_map'])
                           and not all(v>=.99 for v in current['exact_zero_fraction'])
                           and not all(v>=.99 for v in current['exact_one_fraction']))
        candidate_streak=candidate_streak+1 if candidate_quality else 0
        guard=config['collapse_guard']
        exact_collapse=(all(current['constant_map']) or
                        (all(max(z,o)>=guard['exact_saturation_fraction']
                             for z,o in zip(current['exact_zero_fraction'],current['exact_one_fraction'],strict=True))
                         and not any(current['eligible_foreground'])))
        stable_degenerate=(epoch+1>=guard['warmup_epochs'] and global_corr is not None
                           and global_corr>=guard['global_pearson_threshold']
                           and not any(current['eligible_foreground']))
        collapsed=exact_collapse or stable_degenerate
        collapse_reason=('stable_degenerate_monitor' if stable_degenerate else
                         'constant_or_exactly_saturated_maps' if exact_collapse else None)
        with torch.no_grad():
            probe_logits=model.ccam.logits(model.fusion(probe_images.to(device)))
        collapse_streak=collapse_streak+1 if collapsed else 0
        record={'epoch':epoch+1,'sample_count':epoch_samples,'optimizer_updates':len(epoch_loss),
                'mean_ccam_loss':float(np.mean(epoch_loss)),
                'elapsed_seconds':time.monotonic()-epoch_started,
                'rss_peak_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None,
                'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else None,
                'resource_scope':'current_process_session; training_and_monitor; not a pretraining measurement',
                'loss_components':{key:float(np.mean(values)) for key,values in components.items()},
                'gradient_norm_float64_mean':float(np.mean(grad_norms)),
                'gradient_norm_float64_min':float(min(grad_norms)),
                'gradient_nonzero_element_count_mean':float(np.mean(gradient_nonzero_counts)),
                'logit_mean':float(probe_logits.mean().cpu()),
                'logit_std':float(probe_logits.std(unbiased=False).cpu()),
                'monitor_global_pearson':global_corr,'monitor_per_image_pearson':per_image_corr,
                'exact_zero_fraction':current['exact_zero_fraction'],'exact_one_fraction':current['exact_one_fraction'],
                'foreground_fraction':current['foreground_fraction'],
                'background_fraction':current['background_fraction'],'uncertain_fraction':current['uncertain_fraction'],
                'foreground_component_area':current['foreground_component_area'],
                'eligible_foreground':current['eligible_foreground'],'background_nonzero':current['background_nonzero'],
                'constant_map':current['constant_map'],'candidate_eligible':candidate_quality,
                'candidate_consecutive_epochs':candidate_streak,'collapse_consecutive_epochs':collapse_streak,
                'collapse_guard_reason':collapse_reason}
        records.append(record)
        if selected is None and candidate_streak>=3:
            selected={'epoch':epoch+1,'ccam':{name:value.detach().cpu().clone()
                       for name,value in model.ccam.state_dict().items()},
                      'selection_rule':'global_pearson>=0.98_and_all_8_eligible_for_3_consecutive_epochs'}
        atomic_checkpoint(resume_path,{'identity':identity,'ccam':model.ccam.state_dict(),
            'optimizer':optimizer.state_dict(),'epoch':epoch+1,'epoch_records':records,
            'selected_candidate':selected,'candidate_streak':candidate_streak,
            'collapse_streak':collapse_streak,'previous_monitor':current,'rng':rng_state(loader_generator)})
        _atomic_json(out/'metrics.jsonl',records,jsonl=True)
        if sink is not None:sink.log(record)
        previous=current
        if collapse_streak>=3:
            terminal_guard={'status':'stopped_non_success',
                'reason':'three_consecutive_'+str(collapse_reason)+'_epochs',
                'epoch':epoch+1,'collapse_guard':config.get('collapse_guard')}
            atomic_checkpoint(resume_path,{'identity':identity,'ccam':model.ccam.state_dict(),
                'optimizer':optimizer.state_dict(),'epoch':epoch+1,'epoch_records':records,
                'selected_candidate':selected,'candidate_streak':candidate_streak,
                'collapse_streak':collapse_streak,'previous_monitor':current,
                'terminal_guard':terminal_guard,'rng':rng_state(loader_generator)})
            _atomic_json(out/'termination.json',terminal_guard)
            raise RuntimeError('Stabilized CCAM collapse guard stopped this run; resume.pt is current')
    lineage=training_lineage(document,manifest_path,gate_path)
    final=_export_head(out/'ccam.pt',model.ccam.state_dict(),identity=identity,epoch=epochs,
                       lineage=lineage,status='budget_final_not_selected',adaptation_profile=profile)
    selected_origin=None
    if selected is not None:
        selected_origin=_export_head(out/'selected.ccam.pt',selected['ccam'],identity=identity,
            epoch=selected['epoch'],lineage=lineage,status='stability_candidate',adaptation_profile=profile)
    _atomic_json(out/'selection.json',{'rule':selection['rule'],'global_pearson_threshold':.98,
        'required_consecutive_epochs':3,'candidate_eligibility':selection['candidate_eligibility'],
        'budget_epochs':epochs,'stopping_reason':'fixed_budget_exhausted','early_stopping':False,
        'selected_epoch':selected['epoch'] if selected else None,
        'selection_rule':None if selected is None else selected['selection_rule'],
        'status':'stability_candidate' if selected else 'threshold_not_reached',
        'final_checkpoint':{'path':'ccam.pt','sha256':final['checkpoint_sha256']},
        'selected_checkpoint':None if selected_origin is None else {'path':'selected.ccam.pt','sha256':selected_origin['checkpoint_sha256']},
        'scientific_convergence_verified':False,'anatomical_quality_verified':False})
    return final
