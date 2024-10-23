'''
This file implements Embeddings layer
'''
import torch
import torch.nn as nn
class embeddings(nn.Module):

    def __init__(self, vocab_size, embed_dim,):
        super(embeddings, self).__init__()
        self.Embedding = nn.Embedding(vocab_size, embed_dim)

        nn.init.xavier_normal_(self.Embedding.weight)
    
    def forward(self,inp):
        print(torch.max(inp))
        embeds = self.Embedding(inp)
        return embeds