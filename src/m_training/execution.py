# SPDX-License-Identifier: GPL-3.0-or-later
"""Reuse paired features only for the same graph history and the exact reference attention padding."""
from collections import defaultdict
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
def prepare(b,verify=True,canonical_chunk=32):
    length=b['length'];counts=b['counts_list'];offsets=b['offsets_list']
    frames=[(s+j,min(j,2)) for s in b['starts'] for j in range(length)]
    buckets=defaultdict(dict);keys=[];widths=[]
    for g in range(0,len(counts),canonical_chunk):
        sizes=counts[g:g+canonical_chunk];key=(max(sizes),len(sizes))
        pool=buckets[key]
        for i in range(g,g+len(sizes)):
            pool.setdefault(frames[i],i);keys.append((key,frames[i]))
    reps=[];output_index={};groups=[]
    for key,items in buckets.items():
        maximum,reference_groups=key
        items=list(items.items())
        for start in range(0,len(items),reference_groups):
            part=items[start:start+reference_groups]
            for frame,i in part:output_index[(key,frame)]=len(reps);reps.append(i)
            # Pad back to the canonical batch-group and row dimensions in the attention reductions.
            reps.extend([None]*(reference_groups-len(part)));groups.append((reference_groups,maximum))
    rev=np.asarray([output_index[k] for k in keys],dtype=np.int64)
    pts=[];neighbors=[];sizes=[];cursor=0;orig=b['neighbors'].numpy()
    for r in reps:
        if r is None:sizes.append(0);continue
        a,z=offsets[r:r+2];n=z-a;sizes.append(n);pts.append(np.arange(a,z,dtype=np.int64))
        raw=orig[a:z];neighbors.append(np.where(raw>=0,raw-a+cursor,-1));cursor+=n
    def ten(a):
        t=torch.from_numpy(a)
        return t.pin_memory() if b['x'].is_pinned() else t
    return {**b,'rep_index':ten(np.concatenate(pts)),'rep_neighbor':ten(np.concatenate(neighbors)),
            'rep_sizes':ten(np.asarray(sizes,dtype=np.int64)),'rep_reverse':ten(rev),
            'rep_size_list':tuple(sizes),'rep_offsets':tuple(map(int,np.cumsum([0,*sizes],dtype=np.int64))),
            'rep_groups':tuple(groups)}
def bind(base,chunk=32,retain_nodes=65536,graph_checkpoint=False,compact_graph=False):
    assert chunk==32, "Reuse layout preserves canonical attention groups of 32"
    class SamePadM(base):
        def forward(self,b,return_points=False):
            if not self.training or getattr(self.mlla,'canonical_once',False) or 'rep_index' not in b:
                return super().forward(b,return_points)
            assert not return_points and self.mode=='M' and len(self.graph)==2
            # Compute both graph layers in the full original layout, including all window boundaries.
            def encode_graph(raw,previous):
                z=self.encoder(raw)
                for layer in self.graph:z=layer(z,previous)
                return z
            if compact_graph:
                from .compact_graph import encode
                x=encode(self,b)
            else:
                full=checkpoint(encode_graph,b['x'],b['previous'],use_reentrant=False,preserve_rng_state=False) if graph_checkpoint else encode_graph(b['x'],b['previous'])
                x=full.index_select(0,b['rep_index'])
            pos=b['positions'].index_select(0,b['rep_index'])
            out=[];offsets=b['rep_offsets'];g=0;retained_nodes=0
            for n_group,maximum in b['rep_groups']:
                end=g+n_group;lo,hi=offsets[g],offsets[end]
                if hi==lo:g=end;continue
                lens=b['rep_sizes'][g:end]
                starts=torch.cumsum(lens,0)-lens
                grid=torch.arange(maximum,device=x.device)[None,:];valid=grid<lens[:,None]
                pad=torch.where(valid,starts[:,None]+grid,torch.zeros((),device=x.device,dtype=torch.long))
                gids=torch.repeat_interleave(torch.arange(n_group,device=x.device),lens,output_size=hi-lo)
                unpad=gids*maximum+torch.arange(hi-lo,device=x.device)-starts[gids]
                near=b['rep_neighbor'][lo:hi];near=torch.where(near>=0,near-lo,near)
                args=(x[lo:hi],pos[lo:hi],near,pad,valid,unpad,lens)
                if retained_nodes+hi-lo<=retain_nodes:
                    out.append(self.mlla.block(*args));retained_nodes+=hi-lo
                else:out.append(checkpoint(self.mlla.block,*args,use_reentrant=False,preserve_rng_state=False))
                g=end
            h=torch.cat(out,0) if out else x
            counts=b['rep_sizes'];valid=(counts>0).unsqueeze(1)
            mean=torch.segment_reduce(h,reduce='mean',lengths=counts);mx=torch.segment_reduce(h,reduce='max',lengths=counts)
            pooled=torch.cat([torch.where(valid,mean,torch.zeros_like(mean)),torch.where(valid,mx,torch.zeros_like(mx))],dim=1)
            pooled=pooled.index_select(0,b['rep_reverse']).reshape(b['batch_size'],b['length'],128)
            with torch.backends.cudnn.flags(enabled=False):y,_=self.lstm(pooled)
            return self.head(y)
    return SamePadM
