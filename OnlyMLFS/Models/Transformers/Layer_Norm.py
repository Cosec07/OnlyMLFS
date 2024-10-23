'''
Quick implementation of layer norm and residual connections.
'''
import torch
import torch.nn as nn
class AddNorm(nn.Module):

    def __init__(self,embed_dim, eps=1e-6):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self,x,sublayer_op):
        normalized = self.layer_norm(x + sublayer_op)
        return normalized