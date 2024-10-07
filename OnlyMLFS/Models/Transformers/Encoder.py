'''This is the implementation of a single Encoder layer
'''
import torch
import MHA as MHA
import FFN as FFN
import Layer_Norm as LN

class encoder(nn.Module):

    def __init__(self,num_heads,embed_dims):
        super(encoder, self).__init__()
        self.MHA = MHA(num_heads, embed_dims)
        self.LN1 = LN(embed_dims)
        self.FFN = FFN(embed_dims)
        self.LN2 = LN(embed_dims)
    
    def forward(self, inp):
        Attn_op = self.MHA(inp)
        x = self.LN1(inp,Attn_op)
        nn_op = self.FFN(x)
        x = self.LN2(x, nn_op)

        return x

