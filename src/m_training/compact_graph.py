# SPDX-License-Identifier: GPL-3.0-or-later
"""Deduplicate graph nodes by frame pair and window-limited history."""
import numpy as np
import torch
from torch.nn import functional as F

def prepare(b):
    if 'cg_raw_index' in b:return b
    L=b['length'];counts=b['counts_list'];offsets=b['offsets_list']
    frames=[(s+j,j) for s in b['starts'] for j in range(L)]
    maps=[];representatives=[];starts=[]
    for depth in range(3):
        index={};rep=[];pos=[];cursor=0
        for i,(t,j) in enumerate(frames):
            key=(t,min(j,depth))
            if key not in index:index[key]=cursor;rep.append(i);pos.append(cursor);cursor+=counts[i]
        maps.append(index);representatives.append(rep);starts.append(pos)
    prev=b['previous'].numpy();original_to_g2=np.empty(offsets[-1],np.int64)
    raw=np.concatenate([np.arange(offsets[i],offsets[i+1],dtype=np.int64) for i in representatives[0]])
    added={'cg_raw_index':raw}
    for depth in (1,2):
        roots=[];parents=[]
        for i in representatives[depth]:
            t,j=frames[i];n=counts[i];a,z=offsets[i:i+2]
            root=maps[depth-1][(t,min(j,depth-1))]
            roots.append(np.arange(root,root+n,dtype=np.int64))
            if j==0:parents.append(np.full(n,-1,np.int64))
            else:
                source=maps[depth-1][(t-1,min(j-1,depth-1))]
                p=prev[a:z];parents.append(np.where(p>=0,p-offsets[i-1]+source,-1))
        added[f'cg_root{depth}']=np.concatenate(roots);added[f'cg_previous{depth}']=np.concatenate(parents)
    for i,(t,j) in enumerate(frames):
        a,z=offsets[i:i+2];start=maps[2][(t,min(j,2))]
        original_to_g2[a:z]=np.arange(start,start+z-a)
    added['cg_to_rep']=original_to_g2[b['rep_index'].numpy()]
    for k,v in added.items():
        tensor=torch.from_numpy(v)
        added[k]=tensor.pin_memory() if b['x'].is_pinned() else tensor
    return {**b,**added,'cg_node_counts':tuple(sum(counts[i] for i in reps) for reps in representatives)}

def encode(model,b):
    x=model.encoder(b['x'].index_select(0,b['cg_raw_index']))
    for depth,layer in enumerate(model.graph,1):
        previous=b[f'cg_previous{depth}'];roots=b[f'cg_root{depth}']
        message=x[previous.clamp_min(0)]*(previous>=0).to(x.dtype).unsqueeze(1)
        x=F.silu(layer.norm(layer.root(x.index_select(0,roots))+layer.message(message)+layer.bias))
    return x.index_select(0,b['cg_to_rep'])
