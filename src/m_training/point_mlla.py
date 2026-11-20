"""Point-set adaptation of MLLA.

MLLA gate/ELU+1 attention/RoPE/FFN follows LeapLabTHU/MLLA (MIT).
Copyright (c) Microsoft Corporation. See LICENSES/MLLA-MIT.txt.
The four grid depthwise convolutions become independent continuous point kernels.
All attention and local neighbors are isolated to one frame pair in one window.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def spatial_neighbors(uv, stable_keys):
    """Self plus eight nearest other points. Keys only break geometric ties."""
    from scipy.spatial import cKDTree
    uv=np.asarray(uv,dtype=np.float64);n=len(uv)
    result=np.full((n,9),-1,np.int64)
    if not n:return result
    tree=cKDTree(uv);dist,idx=tree.query(uv,k=min(n,10),workers=1)
    dist=dist.reshape(n,-1);idx=idx.reshape(n,-1)
    squared=np.sum((uv[idx]-uv[:,None,:])**2,axis=-1)
    squared=np.where(idx==np.arange(n)[:,None],np.inf,squared)
    order=np.lexsort((np.asarray(stable_keys)[idx],squared),axis=1)
    nearest=np.take_along_axis(idx,order,axis=1)[:,:min(8,n-1)]
    result[:,0]=np.arange(n);result[:,1:1+nearest.shape[1]]=nearest
    # Only geometric ties crossing the query boundary need an expanded query.
    tied=np.flatnonzero(np.isclose(dist[:,-1],dist[:,-2],rtol=0,atol=1e-12)) if n>9 else []
    for i in tied:
        candidates=np.asarray(tree.query_ball_point(uv[i],float(dist[i,-1])+1e-10),np.int64)
        candidates=candidates[candidates!=i]
        d2=np.sum((uv[candidates]-uv[i])**2,axis=1)
        result[i,1:]=candidates[np.lexsort((np.asarray(stable_keys)[candidates],d2))][:8]
    return result


class PointDepthwise(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.kernel=nn.Sequential(nn.Linear(2,16),nn.SiLU(),nn.Linear(16,width))
        self.bias=nn.Parameter(torch.zeros(width))

    def forward(self,x,delta,neighbors,mask):
        weights=self.kernel(delta)
        messages=weights*x[neighbors]*mask.unsqueeze(-1)
        return messages.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)+self.bias


class PointMLLA(nn.Module):
    def __init__(self,width=64,heads=4,mlp_ratio=4,frame_chunk_size=32,checkpoint_enabled=True):
        super().__init__()
        assert width==64 and heads==4 and mlp_ratio==4
        self.width=width;self.heads=heads
        self.frame_chunk_size=frame_chunk_size;self.checkpoint_enabled=checkpoint_enabled
        self.cpe1=PointDepthwise(width);self.norm1=nn.LayerNorm(width)
        self.in_proj=nn.Linear(width,width);self.act_proj=nn.Linear(width,width)
        self.dwc=PointDepthwise(width);self.qk=nn.Linear(width,width*2)
        self.lepe=PointDepthwise(width);self.out_proj=nn.Linear(width,width)
        self.cpe2=PointDepthwise(width);self.norm2=nn.LayerNorm(width)
        self.mlp=nn.Sequential(nn.Linear(width,width*mlp_ratio),nn.GELU(),nn.Linear(width*mlp_ratio,width))
        self.register_buffer('frequencies',10000**(-torch.arange(width//4,dtype=torch.float32)/(width//4)),persistent=False)

    def rope(self,x,positions):
        angles=(positions.unsqueeze(-1)*self.frequencies).flatten(1)
        z=x.reshape(len(x),-1,2);c=angles.cos();s=angles.sin()
        return torch.stack((z[...,0]*c-z[...,1]*s,z[...,0]*s+z[...,1]*c),-1).flatten(1)

    def attention(self,x,positions,pad_index,pad_valid,unpad,counts,delta,neighbors,neighbor_mask):
        q,k=self.qk(x).chunk(2,dim=-1);q=F.elu(q)+1;k=F.elu(k)+1
        qr=self.rope(q,positions);kr=self.rope(k,positions)
        groups,maximum=pad_index.shape;d=self.width//self.heads
        def padded(t,mask=False):
            y=t[pad_index]
            if mask:y=y*pad_valid.unsqueeze(-1)
            return y.reshape(groups,maximum,self.heads,d).transpose(1,2)
        qh=padded(q);kh=padded(k,True);qh_rope=padded(qr);kh_rope=padded(kr,True);v=padded(x,True)
        count=counts.to(x.dtype).clamp_min(1).reshape(groups,1,1,1)
        normalizer=(qh@(kh.sum(-2,keepdim=True)/count).transpose(-2,-1)+1e-6).reciprocal()
        scale=count.rsqrt();kv=(kh_rope.transpose(-2,-1)*scale)@(v*scale)
        y=(qh_rope@kv*normalizer).transpose(1,2).reshape(groups*maximum,self.width)[unpad]
        return y+self.lepe(x,delta,neighbors,neighbor_mask)

    def block(self,x,positions,neighbor_index,pad_index,pad_valid,unpad,counts):
        mask=(neighbor_index>=0).to(x.dtype);neighbors=neighbor_index.clamp_min(0)
        delta=positions[neighbors]-positions[:,None,:]
        # Mask unused slots before the kernel, as well as before aggregation.
        delta=delta*mask.unsqueeze(-1)
        x=x+self.cpe1(x,delta,neighbors,mask)
        z=self.norm1(x);gate=F.silu(self.act_proj(z))
        z=F.silu(self.dwc(self.in_proj(z),delta,neighbors,mask))
        x=x+self.out_proj(self.attention(z,positions,pad_index,pad_valid,unpad,counts,delta,neighbors,mask)*gate)
        x=x+self.cpe2(x,delta,neighbors,mask)
        return x+self.mlp(self.norm2(x))

    def forward(self,x,batch):
        if not len(x):return x
        counts=batch['counts_list'];offsets=batch['offsets_list'];outputs=[]
        for g in range(0,len(counts),self.frame_chunk_size):
            end=min(g+self.frame_chunk_size,len(counts));lo,hi=offsets[g],offsets[end]
            if lo==hi:continue
            sizes=counts[g:end];maximum=max(sizes)
            starts=batch['offsets'][g:end]-lo;lengths=batch['counts'][g:end]
            grid=torch.arange(maximum,device=x.device)[None,:]
            valid=grid<lengths[:,None]
            pad=torch.where(valid,starts[:,None]+grid,torch.zeros((),device=x.device,dtype=torch.long))
            group_ids=torch.repeat_interleave(torch.arange(end-g,device=x.device),lengths,output_size=hi-lo)
            unpad=group_ids*maximum+torch.arange(hi-lo,device=x.device)-starts[group_ids]
            neighbor=batch['neighbors'][lo:hi]
            neighbor=torch.where(neighbor>=0,neighbor-lo,neighbor)
            args=(x[lo:hi],batch['positions'][lo:hi],neighbor,pad,valid,unpad,lengths)
            if self.training and torch.is_grad_enabled() and self.checkpoint_enabled:
                y=checkpoint(self.block,*args,use_reentrant=False,preserve_rng_state=False)
            else:y=self.block(*args)
            outputs.append(y)
        return torch.cat(outputs,0)
