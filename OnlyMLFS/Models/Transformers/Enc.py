'''This is the implementation of a single Encoder layer
'''
import torch
import torch.nn as nn
import OnlyMLFS.Models.Transformers.MHA as MHA
import OnlyMLFS.Models.Transformers.FFN as PFFN
import OnlyMLFS.Models.Transformers.Layer_Norm as AddNorm

class encoder(nn.Module):

    def __init__(self,num_heads,embed_dims, dropout_rate = 0.1):
        super(encoder, self).__init__()
        self.MHA = MHA.MHA(num_heads, embed_dims, dropout_rate)

        self.LN1 = AddNorm.AddNorm(embed_dims)
        self.FFN = PFFN.PFFN(embed_dims)
        self.LN2 = AddNorm.AddNorm(embed_dims)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(self, inp, mask=None):
        Attn_op = self.MHA(inp, mask)
        x = self.LN1(inp,self.dropout1(Attn_op))

        nn_op = self.FFN(x)
        x = self.LN2(x, self.dropout2(nn_op))

        return x

