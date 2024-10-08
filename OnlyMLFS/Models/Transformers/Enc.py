'''This is the implementation of a single Encoder layer
'''
import torch
import torch.nn as nn
import OnlyMLFS.Models.Transformers.MHA as MHA
import OnlyMLFS.Models.Transformers.FFN as FFN
import OnlyMLFS.Models.Transformers.Layer_Norm as LN

class encoder(nn.Module):

    def __init__(self,num_heads,embed_dims):
        super(encoder, self).__init__()
        self.MHA = MHA.MHA(num_heads, embed_dims)
        self.LN1 = LN.AddNorm(embed_dims)
        self.FFN = FFN.PFFN(embed_dims)
        self.LN2 = LN.AddNorm(embed_dims)
        self.embed_dims = embed_dims
        self.em = nn.Embedding(10, self.embed_dims)
    
    def forward(self, inp):
        Attn_op = self.MHA(inp)
        x = self.LN1(inp,Attn_op)
        nn_op = self.FFN(x)
        x = self.LN2(x, nn_op)

        return x

