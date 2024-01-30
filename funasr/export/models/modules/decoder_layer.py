#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn


class DecoderLayerSANM(nn.Module):

    def __init__(
        self,
        model
    ):
        super().__init__()
        self.self_attn = model.self_attn
        self.src_attn = model.src_attn
        self.feed_forward = model.feed_forward
        self.norm1 = model.norm1
        self.norm2 = model.norm2 if hasattr(model, 'norm2') else None
        self.norm3 = model.norm3 if hasattr(model, 'norm3') else None
        self.size = model.size
        self.reserve_attn = False


    def forward(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):

        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn is not None:
            tgt = self.norm2(tgt)
            x, cache = self.self_attn(tgt, tgt_mask, cache=cache)
            x = residual + x

        if self.src_attn is not None:
            residual = x
            x = self.norm3(x)
            if self.reserve_attn:
                x_src_attn, attn_mat = self.src_attn(x, memory, memory_mask, ret_attn=True)
                self.attn_mat.append(attn_mat)
            else:
                x_src_attn = self.src_attn(x, memory, memory_mask, ret_attn=False)

            x = residual + x_src_attn


        return x, tgt_mask, memory, memory_mask, cache

    def get_attn_mat(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn is not None:
            tgt = self.norm2(tgt)
            x, cache = self.self_attn(tgt, tgt_mask, cache=cache)
            x = residual + x

        residual = x
        x = self.norm3(x)
        x_src_attn, attn_mat = self.src_attn(x, memory, memory_mask, ret_attn=True)
        return attn_mat


class DecoderLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.self_attn = model.self_attn
        self.src_attn = model.src_attn
        self.feed_forward = model.feed_forward
        self.norm1 = model.norm1
        self.norm2 = model.norm2
        self.norm3 = model.norm3
    
    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        residual = tgt
        tgt = self.norm1(tgt)
        tgt_q = tgt
        tgt_q_mask = tgt_mask
        x = residual + self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)

        residual = x
        x = self.norm2(x)
        
        x = residual + self.src_attn(x, memory, memory, memory_mask)

        residual = x
        x = self.norm3(x)
        x = residual + self.feed_forward(x)

        return x, tgt_mask, memory, memory_mask
