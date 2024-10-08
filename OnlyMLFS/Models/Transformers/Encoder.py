'''
This is the implementation of a Encoder Network
'''
import OnlyMLFS.Models.Transformers.Enc as ENC
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, num_layers,num_heads,embed_dims):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.encoders = nn.ModuleList([ENC.encoder(num_heads,embed_dims) for _ in range(self.num_layers)])
    
    def forward(self,x):
        for layer in self.encoders:
            x = layer(x)
        return x

