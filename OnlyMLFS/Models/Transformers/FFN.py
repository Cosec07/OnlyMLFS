'''
This is the Position-wise feed forward neural network implementation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class PFFN(nn.Module):

    def __init__(self,embedding_dim):
        super(PFFN,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = 4 * self.embedding_dim
        self.L1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.act = nn.ReLU()
        self.L2 = nn.Linear(self.hidden_dim, self.embedding_dim)
    
    def forward(self,x):
        x = self.L1(x)
        x = self.act(x)
        x = self.L2(x)
        return x
        

