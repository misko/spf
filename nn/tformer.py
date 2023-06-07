import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np

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
