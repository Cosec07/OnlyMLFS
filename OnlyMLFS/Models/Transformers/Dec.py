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
        self.self_attn = MHA.MHA(num_heads,embed_dim, dropout_rate)
        self.cross_attn = MHA.MHA(num_heads,embed_dim, dropout_rate)
        self.FFN = PFFN.PFFN(embed_dim)

        self.LN1 = AddNorm.AddNorm(embed_dim)
        self.LN2 = AddNorm.AddNorm(embed_dim)
        self.LN3 = AddNorm.AddNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
    
    def forward(self,x,enc_out, tgt_mask=None, src_mask = None):
      if tgt_mask is None:
        tgt_mask = self.self_attn.generate_causal_mask(x.size(1))
        tgt_mask = tgt_mask.to(x.device)

        masked_attn = self.self_attn(x, mask=tgt_mask)
        x= self.LN1(x,self.dropout1(masked_attn))

        cross_attn = self.cross_attn(x, enc_out, enc_out, mask = src_mask)
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