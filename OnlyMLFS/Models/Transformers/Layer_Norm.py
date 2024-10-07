'''
Quick implementation of layer norm and residual connections.
'''
import torch
import torch.nn as nn
class AddNorm(nn.Module):

    def __init__(self,embed_dim):
        super(AddNorm, self).__init__()
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self,x,sublayer_op):
        x = x + sublayer_op
        x = self.layer_norm(x)
        return x