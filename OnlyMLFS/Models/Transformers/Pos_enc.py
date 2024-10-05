'''
This is the implementation of Positional Encoding using Sine and Cosine functions to encode the sequencential data.
'''
import torch
import torch.nn as nn
import math

class POS_ENC(nn.Module):

    def __init__(self,embed_dim,seq_len):
        super(POS_ENC, self). __init__()
        self.emded_dim = embed_dim
        self.seq_len = seq_len
        self.pe = torch.zeros(self.seq_len, self.embed_dim)
        #calculate Positional Encodings
        for pos in range(self.seq_len):
            for i in range(0,self.embed_dim,2):
                self.pe[pos,i] = math.sin(pos/(10000 ** ((2*i)/self.embed_dim)))
                if i+1 < self.embed_dim:
                    self.pe[pos,i+1] = math.cos(pos/(10000 ** ((2*i)/self.embed_dim)))
        self.pe = self.pe.unsqueeze(0)
    
    def forward(self,x):
        x = x + self.pe[:, :x.size(1), :]
        return x


