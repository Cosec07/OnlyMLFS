'''
Combines other components to create a Transformer network
'''
import torch
import torch.nn as nn
import OnlyMLFS.Models.Transformers.embeddings as embeddings
import OnlyMLFS.Models.Transformers.Pos_enc as Pos_enc
import OnlyMLFS.Models.Transformers.Encoder as Encoder
import OnlyMLFS.Models.Transformers.Decoder as Decoder

class Transformer(nn.Module):

    def __init__(self,vocab_size, embed_dims, num_heads, num_layers,max_seq_length,dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.src_embedding = embeddings.embeddings(vocab_size,embed_dims)
        self.tgt_embedding = embeddings.embeddings(vocab_size,embed_dims)

        self.positional_encoding = Pos_enc.POS_ENC(embed_dims, max_seq_length,dropout_rate)

        self.encoder = Encoder.Encoder(num_layers, num_heads, embed_dims)

        self.decoder = Decoder.Decoder(num_layers, num_heads, embed_dims)

        self.output_projection = nn.Linear(embed_dims,vocab_size)
    
    def padding_mask(self, seq):
        return (seq != 0).unsqueeze(1).unsqueeze(2)
    
    def forward(self,src, tgt):
        src_mask = self.padding_mask(src)
        tgt_mask = self.padding_mask(tgt)

        src_embeds = self.embedding(src)
        tgt_embeds = self.embedding(tgt)

        src_encodings = self.positional_encoding(src_embeds)
        tgt_encodings = self.positional_encoding(tgt_embeds)

        encoder_out = self.encoder(src_embeds, src_mask)
        decoder_out = self.decoder(tgt_embeds, encoder_out, tgt_mask, src_mask)


        out = self.output_projection(decoder_out)

        return out



