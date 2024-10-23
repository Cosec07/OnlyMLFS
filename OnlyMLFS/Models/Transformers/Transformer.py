'''
Combines other components to create a Transformer network
'''
import torch
import torch.nn as nn
import OnlyMLFS.Models.Transformers.embeddings as embeddings
import OnlyMLFS.Models.Transformers.Pos_enc as Pos_enc
import OnlyMLFS.Models.Transformers.Encoder as Encoder

class Transformer(nn.Module):

    def __init__(self,vocab_size, embed_dims, num_heads, num_layers,max_seq_length,dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.embedding = embeddings.embeddings(vocab_size,embed_dims)

        self.positional_encoding = Pos_enc.POS_ENC(embed_dims, max_seq_length)

        self.encoder = Encoder.Encoder(num_layers, num_heads, embed_dims)

        self.dropout = nn.Dropout(dropout_rate)

        self.output_projection = nn.Linear(embed_dims,vocab_size)
    
    def forward(self,inp):
        x = self.embedding(inp)

        x = self.positional_encoding(x)

        x = self.dropout(x)

        encoder_out = self.encoder(x)

        out = self.output_projection(encoder_out)

        return out
    
    def generate_attention_mask(self,sz):
        mask = (torch.triu(torch.ones(sz,sz) == 1).transpose(0,1))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



