# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import os
import json


class BaseModel(nn.Module):
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def save(self, save_dir, fname="model", metrics=None):
        model_state = os.path.join(save_dir, "{}.pt".format(fname))
        torch.save(self.state_dict(), model_state)
        with open(os.path.join(save_dir, "{}_config.json".format(fname)), "w") as wf:
            json.dump(self.init_params, wf, indent=2)
        if metrics is not None:
           with open(os.path.join(save_dir, "{}_metrics.json".format(fname)), "w") as wf:
            json.dump(metrics, wf, indent=2) 
    
    @staticmethod
    def load(model_cls, load_dir, fname):
        with open(os.path.join(load_dir, "{}_config.json".format(fname))) as rf:
            init_params = json.load(rf)
        model = model_cls(**init_params)
        model.load_state_dict(torch.load(os.path.join(load_dir, "{}.pt".format(fname))))
        return model
    
    def loss(self, p, y):
        # Apply label smoothing, if any
        if hasattr(self, "ls"):
            if self.ls > 0.:
                y = ((1.0-self.ls)*y) + (1.0/y.size(1))
        loss = nn.BCELoss()(p, y)
        return loss


class LowFER(BaseModel):
    
    def __init__(self, ne, nr, de, dr, k=10, d_in=0.2, d_h1=0.2, d_h2=0.3, ls=0., reg=0.):
        super(LowFER, self).__init__()
        
        # Entities and relations embeddings
        self.E = nn.Embedding(ne, de, padding_idx=0)
        self.R = nn.Embedding(nr, dr, padding_idx=0)
        # Low-rank matrices
        self.U = nn.Parameter(torch.from_numpy(np.random.uniform(-1, 1, (de, k * de))).float())
        self.V = nn.Parameter(torch.from_numpy(np.random.uniform(-1, 1, (dr, k * de))).float())
        # Dropout layers
        self.d_in = nn.Dropout(d_in)
        self.d_h1 = nn.Dropout(d_h1)
        self.d_h2 = nn.Dropout(d_h2)
        # Batch normalization layers
        self.bn_in = nn.BatchNorm1d(de)
        self.bn_out = nn.BatchNorm1d(de)
        
        self.ne, self.de = ne, de
        self.nr, self.dr = nr, dr
        self.k = k
        self.ls = ls
        self.reg = reg # 0.0005
        self.init_params = {
            "ne": ne, "nr": nr, "de": de, "dr": dr, "k": k,
            "d_in": d_in, "d_h1": d_h1, "d_h2": d_h2, "ls": ls
        }
        self.init()
    
    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)
    
    def normalization(self, x, dim=-1, power=True, l2=True, eps=1e-12):
        if power:
            x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + eps))
        if l2:
            x = nn.functional.normalize(x, p=2, dim=dim)
        return x
    
    def mfb(self, h, r):
        """Multi-modal factorized bilinear pooling.
        
        Parameters:
        -----------
            h : (B, de)
            r : (B, dr)
        
        """
        x = torch.mm(h, self.U) # B x de * de x k*de -> B x k*de
        y = torch.mm(r, self.V) # B x dr * dr x k*de -> B x k*de
        z = x * y
        z = self.d_h1(z)
        z = z.view(-1, self.de, self.k) # B x de x k
        z = z.sum(-1) # B x de
        z = self.normalization(z)
        return z
    
    def forward(self, h_idx, r_idx, labels=None):
        h = self.E(h_idx)
        h = self.d_in(self.bn_in(h))
        r = self.R(r_idx)
        
        # Apply MFB layer
        x = self.mfb(h, r)
        x = self.d_h2(self.bn_out(x))
        
        # Score
        s = x.matmul(self.E.weight.transpose(0, 1)) # B x de * de x ne -> B x ne
        s = torch.sigmoid(s)
        outputs = (s,)
        
        if labels is not None:
            loss = self.loss(s, labels)
            if self.reg > 0.:
                loss += self.reg *self.l2_regularize(h, r)
            outputs += (loss,)
        
        return outputs
    
    def l2_regularize(self, h, r):
        reg = (h ** 2).mean() + (self.E.weight ** 2).mean() + (r ** 2).mean() / 3
        return reg
