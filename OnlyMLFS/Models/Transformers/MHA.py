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
        
        self.qkv_proj = nn.Linear(embed_dims, 3 * embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)

        self.dropout = nn.Dropout(dropout_rate)

        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self,inp, mask=None):
        batch_size, seq_len, _ = inp.shape
        #Linear Projections of Q,K,V
        qkv = self.qkv_proj(inp)
        
        #Split heads
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_head, self.head_dim)
        
        qkv = qkv.permute(2,0,3,1,4)
        
        q,k,v = qkv[0], qkv[1], qkv[2]
        #Perfrom SDPA
        scores = torch.matmul(q, k.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        #Apply attention mask if necessary
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        #Attention weights after dropout
        attention = self.dropout(F.softmax(scores, dim=-1))
        #Generate Context Vectors
        inp = torch.matmul(attention, v)
        #Concat heads and Final Layer Linear Projection
        inp = inp.transpose(1,2).contiguous().view(batch_size, seq_len, self.embed_dims)
        out = self.out_proj(inp)

        return out