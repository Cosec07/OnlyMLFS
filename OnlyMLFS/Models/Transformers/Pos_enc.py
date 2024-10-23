'''
This is the implementation of Positional Encoding using Sine and Cosine functions to encode the sequencential data.
'''
import torch
import torch.nn as nn
import math

class POS_ENC(nn.Module):

    def __init__(self,embed_dim,max_seq_len, dropout_rate = 0.1):
        super(POS_ENC, self). __init__()
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0)/ embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self,x):
        x = x + self.pe[:, :x.size(1), :]
        return x


