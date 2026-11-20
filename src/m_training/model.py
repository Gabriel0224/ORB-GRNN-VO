# SPDX-License-Identifier: GPL-3.0-or-later
import torch
from torch import nn

class TrajectoryLayer(nn.Module):
    """At most one incoming temporal edge per node."""
    def __init__(self,width=64):
        super().__init__()
        self.root=nn.Linear(width,width,bias=False)
        self.message=nn.Linear(width,width,bias=False)
        self.bias=nn.Parameter(torch.zeros(width))
        self.norm=nn.LayerNorm(width)
    def forward(self,x,previous):
        if len(x)==0:return x
        message=x[previous.clamp_min(0)]*(previous>=0).to(x.dtype).unsqueeze(1)
        return torch.nn.functional.silu(self.norm(self.root(x)+self.message(message)+self.bias))

class PointSetModel(nn.Module):
    def __init__(self, frame_chunk_size=32, checkpoint_enabled=True):
        super().__init__()
        self.mode='M'
        self.encoder=nn.Sequential(nn.Linear(6,64),nn.LayerNorm(64),nn.SiLU(),
                                   nn.Linear(64,64),nn.LayerNorm(64),nn.SiLU())
        self.lstm=nn.LSTM(128,128,num_layers=2,batch_first=True,dropout=.2)
        self.head=nn.Linear(128,6)
        self.graph=nn.ModuleList([TrajectoryLayer(),TrajectoryLayer()])
        from .point_mlla import PointMLLA
        self.mlla=PointMLLA(frame_chunk_size=frame_chunk_size,checkpoint_enabled=checkpoint_enabled)
    def forward(self,batch,return_points=False):
        x=self.encoder(batch['x'])
        for layer in self.graph:x=layer(x,batch['previous'])
        x=self.mlla(x,batch)
        counts=batch['counts'];valid=(counts>0).unsqueeze(1)
        mean=torch.segment_reduce(x,reduce='mean',lengths=counts)
        maximum=torch.segment_reduce(x,reduce='max',lengths=counts)
        pooled=torch.cat([torch.where(valid,mean,torch.zeros_like(mean)),
                          torch.where(valid,maximum,torch.zeros_like(maximum))],dim=1)
        pooled=pooled.reshape(batch['batch_size'],batch['length'],128)
        with torch.backends.cudnn.flags(enabled=False):y,_=self.lstm(pooled)
        pred=self.head(y)
        return (pred,x,pooled) if return_points else pred
