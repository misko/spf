import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from functools import cache

@cache
def get_grid(width):
        _xy=torch.arange(width)
        _x,_y=torch.meshgrid(_xy,_xy)
        return torch.cat([_y[None],_x[None]]).transpose(0,2)

def detector_positions_to_distance(detector_positions,width):
        diffs=get_grid(ds.args.width)[None,None]-detector_positions[:,:,None,None]
        return (torch.sqrt(torch.pow(diffs, 2).sum(axis=4))/width)[:,:,None] # batch, snapshot, 1,x ,y 

def detector_positions_to_theta_grid(detector_positions,width):
        diffs=get_grid(ds.args.width)[None,None]-detector_positions[:,:,None,None]
        return (torch.atan2(diffs[...,1],diffs[...,0]))[:,:,None] # batch, snapshot,1, x ,y `

class TransformerModel(nn.Module):

    def __init__(self, d_radio:int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, n_outputs=4):
        super().__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.n_outputs=n_outputs
        self.linear_out = nn.Linear(d_model, n_outputs)
        assert( d_model>d_radio)
        self.linear_in = nn.Linear(d_radio, d_model-d_radio)
        self.d_model=d_model
        self.d_radio=d_radio

    def forward(self, src: Tensor) -> Tensor:
        #output = self.transformer_encoder(src)
        output = self.transformer_encoder(torch.cat([src,self.linear_in(src)],axis=2)) #/np.sqrt(self.d_radio))
        output = self.linear_out(output)/np.sqrt(self.d_model)
        return output

class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        #num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):  
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        #self.positional_encoder = PositionalEncoding(
        #    dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        #)
        #self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, 2)
        self.enc_tgt = nn.Linear(2,dim_model)


    def forward(
        self,
        src,
        tgt,
    ):  
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)
        tgt=self.enc_tgt(tgt)
        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = src * math.sqrt(self.dim_model)
        tgt = tgt * math.sqrt(self.dim_model)
        #src = self.positional_encoder(src)
        #tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt)
        out = self.out(transformer_out).permute(1,0,2)

        return out
