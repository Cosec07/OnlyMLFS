'''
This is the implementation of the Decoder stack
'''
import torch
import torch.nn as nn
import OnlyMLFS.Models.Transformers.Dec as decoder

class Decoder(nn.Module):
    
    def __init__(self, num_layers,num_heads,embed_dims, dropout_rate = 0.1):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([
            decoder.decoder(num_heads, embed_dims, dropout_rate) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dims)
    
    def forward(self,x, enc_out, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm(x)

