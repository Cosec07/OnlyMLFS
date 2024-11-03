'''
this is the Implementation for multi head attention
'''
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self,num_heads,embed_dims, dropout_rate = 0.1):
        super(MHA,self).__init__()
        assert embed_dims % num_heads == 0, "embed_dims must be divisible by num_heads"

        self.num_head = num_heads
        self.embed_dims = embed_dims
        self.head_dim = self.embed_dims // num_heads
        
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)

        self.dropout = nn.Dropout(dropout_rate)

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def generate_causal_mask(self,size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask
    
    def forward(self,q_inp, k_inp=None, v_inp=None, mask=None):
        if k_inp is None:
            k_inp = q_inp
        if v_inp is None:
            v_inp = q_inp

        batch_size, seq_len, _ = q_inp.shape
        #Linear Projections of Q,K,V
        q = self.q_proj(q_inp)
        k = self.k_proj(k_inp)
        v = self.v_proj(v_inp)
        
        #Split heads
        q = q.reshape(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        k = k.reshape(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        v = v.reshape(batch_size, seq_len, self.num_head, self.head_dim).transpose(1,2)
        
        #Perfrom SDPA
        scores = torch.matmul(q, k.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        #Apply attention mask if necessary
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        #Attention weights after dropout
        attention = self.dropout(F.softmax(scores, dim=-1))
        #Generate Context Vectors
        out = torch.matmul(attention, v)
        #Concat heads and Final Layer Linear Projection
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, self.embed_dims)
        out = self.out_proj(out)
        return out