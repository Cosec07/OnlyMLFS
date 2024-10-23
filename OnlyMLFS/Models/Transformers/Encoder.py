'''
This is the implementation of a Encoder Network
'''
import OnlyMLFS.Models.Transformers.Enc as encoder
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, num_layers,num_heads,embed_dims, dropout_rate = 0.1):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            encoder.encoder(num_heads, embed_dims, dropout_rate) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dims)
    
    def forward(self,x, mask = None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

