"""HAGO-inspired optimizer control with declared task-balance sign policy."""
from __future__ import annotations
import torch


class HAGO:
    def __init__(self,model,optimizer,*,warmup_epochs=10,variance_threshold=1.,dominance_threshold=.1,
                 logistic_k=1.,balance='suppress_dominant',entropy_threshold=.1,noise_std=0.,seed=42):
        if balance not in {'suppress_dominant','paper_logistic'}: raise ValueError('Unknown HAGO balance convention')
        if min(warmup_epochs,variance_threshold,dominance_threshold,logistic_k,entropy_threshold,noise_std)<0:
            raise ValueError('Nonnegative HAGO controls required')
        self.model=model;self.optimizer=optimizer;self.names=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        self.warmup=warmup_epochs;self.vth=variance_threshold;self.dth=dominance_threshold
        self.k=logistic_k;self.balance=balance;self.eth=entropy_threshold;self.noise=noise_std;self.seed=seed;self.step_number=0
    def step(self,segmentation,auxiliary,coverage_logits,epoch):
        params=[p for _,p in self.names];self.optimizer.zero_grad(set_to_none=True)
        if epoch<self.warmup:
            (segmentation+auxiliary).backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params): raise ValueError('Nonfinite warmup gradient')
            self.optimizer.step();return {'warmup':True,'segmentation_weight':.5,'noise_applied':False}
        sg=torch.autograd.grad(segmentation,params,retain_graph=True,allow_unused=True)
        ag=torch.autograd.grad(auxiliary,params,allow_unused=True)
        ns=torch.sqrt(sum((g.square().sum() for g in sg if g is not None),segmentation.new_zeros(())))
        na=torch.sqrt(sum((g.square().sum() for g in ag if g is not None),segmentation.new_zeros(())))
        delta=ns-na;sign=-1 if self.balance=='suppress_dominant' else 1
        weight=torch.sigmoid(sign*self.k*delta) if delta.abs()>self.dth else delta.new_tensor(.5)
        gradients=[]
        # Group weight+bias tensors by named layer, not by one global clipping norm.
        groups={}
        for i,(name,p) in enumerate(self.names):
            s=torch.zeros_like(p) if sg[i] is None else sg[i]
            a=torch.zeros_like(p) if ag[i] is None else ag[i]
            g=weight*s+(1-weight)*a
            if not torch.isfinite(g).all(): raise ValueError('Nonfinite HAGO task gradient')
            gradients.append(g);groups.setdefault(name.rsplit('.',1)[0],[]).append(i)
        for indices in groups.values():
            flat=torch.cat([gradients[i].reshape(-1) for i in indices])
            norm=flat.norm();variance=flat.var(unbiased=False)
            scale=1/(1+variance) if variance>self.vth else variance.new_tensor(1.)
            for i in indices:params[i].grad=gradients[i]/(norm+1e-8)*scale
        eps=torch.finfo(coverage_logits.dtype).eps
        probabilities=coverage_logits.detach().sigmoid().clamp(eps,1-eps)
        entropy=(-(probabilities*probabilities.log()+(1-probabilities)*(1-probabilities).log())).mean()
        noise_applied=bool(self.noise>0 and entropy<self.eth)
        self.optimizer.step()
        if noise_applied:
            with torch.no_grad():
                for index,(name,p) in enumerate(self.names):
                    if any(key in name for key in ('segmentation','coverage','cross')):
                        generator=torch.Generator(device=p.device).manual_seed(self.seed+self.step_number*len(self.names)+index)
                        noise=torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=generator)
                        lr=self.optimizer.param_groups[0]['lr'];p.add_(noise,alpha=lr*self.noise)
        self.step_number+=1
        return {'warmup':False,'segmentation_weight':float(weight.detach()),'entropy':float(entropy),
                'noise_applied':noise_applied,'balance':self.balance}
