'''
This is the Position-wise feed forward neural network implementation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class PFFN(nn.Module):

    def __init__(self,embedding_dim, expansion_factor = 4, dropout_rate = 0.1):
        super(PFFN,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = expansion_factor * self.embedding_dim

        self.Linear1 = nn.Linear(embedding_dim, self.hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.Linear2 = nn.Linear(self.hidden_dim, embedding_dim)
        self.act = nn.ReLU()

        nn.init.xavier_uniform_(self.Linear1.weight)
        nn.init.xavier_uniform_(self.Linear2.weight)
        nn.init.zeros_(self.Linear1.bias)
        nn.init.zeros_(self.Linear2.bias)
    
    def forward(self,x):
        x = self.Linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        return x
        

