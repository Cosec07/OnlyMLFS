'''
This is the implementation for a single Decoder layer
'''
import torch
import torch.nn as nn
import OnlyMLFS.Models.Transformers.MHA as MHA
import OnlyMLFS.Models.Transformers.FFN as PFFN
import OnlyMLFS.Models.Transformers.Layer_Norm as AddNorm

class decoder(nn.Module):
    
    def __init__(self,num_heads,embed_dim, dropout_rate = 0.1):
        super(decoder, self).__init__()
        self.MHA = MHA.MHA(num_heads,embed_dim, dropout_rate)

        self.FFN = PFFN.PFFN(embed_dim)

        self.LN1 = AddNorm.AddNorm(embed_dim)
        self.LN2 = AddNorm.AddNorm(embed_dim)
        self.LN3 = AddNorm.AddNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
    
    def forward(self,x,enc_out):
        masked_attn = MMHA(x)
        x= self.LN1(x,self.dropout1(masked_attn))

        cross_attn = self.MHA(enc_out)
        x = self.LN2(x, self.dropout2(cross_attn))

        nn_op = self.FFN(x)
        x = self.LN3(x,self.dropout3(nn_op))
        
        return x




'''
Input
  ↓
[Masked Self-Attention]  ← (can only look at previous tokens)
  ↓
Add & Norm
  ↓
[Cross-Attention]        ← (looks at encoder output)
  ↓
Add & Norm
  ↓
[Feed Forward]
  ↓
Add & Norm
  ↓
Output
'''