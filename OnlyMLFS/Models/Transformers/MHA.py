'''
this is the Implementation for multi head attention
'''
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self,num_heads,embed_dims, vocab_size = 10):
        super(MHA,self).__init__()
        self.num_head = num_heads
        self.embed_dims = embed_dims
        self.head_dim = self.embed_dims // num_heads
        self.embedding = nn.Embedding(vocab_size, self.embed_dims)
        self.W_q = nn.Linear(embed_dims,embed_dims)
        self.W_k = nn.Linear(embed_dims,embed_dims)
        self.W_v = nn.Linear(embed_dims,embed_dims)
        self.W_o = nn.Linear(embed_dims,embed_dims)
    
    def embeddings(self,inp):
                embeds = self.embedding(inp)
                return embeds,embeds,embeds
    
    def split_heads(self,X):
            batch_size, seq_len, _ = X.shape
            X = X.view(batch_size, seq_len, self.num_head, self.head_dim)
            X = X.permute(0,2,1,3)
            return X
            
    def SDPA(self,Q,K,V):
        kd = torch.tensor(K.shape[-1],dtype=torch.float32)
        con_vecs = torch.matmul((F.softmax(torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(kd), dim=-1)),V)    
        return con_vecs

    def concat_heads(self,X):
        batch_size, num_heads, seq_len, head_dim = X.shape
        X = X.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return X
    
    def forward(self,inp):
        #Embeddings
        Q,K,V = self.embeddings(inp)
        #Linear Projections of Q,K,V
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        #Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        #Perfrom SDPA
        con_vecs = self.SDPA(Q,K,V)
        #concat heads
        con_vec = self.concat_heads(con_vecs)
        #Final Layer Linear Projection
        out = self.W_o(con_vec)


        return out