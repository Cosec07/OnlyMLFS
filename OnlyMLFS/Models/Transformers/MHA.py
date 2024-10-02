'''
this is the Implementation for multi head attention
'''
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self,num_heads,embed_dims):
        super(MHA,self).__init__()
        self.num_head = num_heads
        self.embed_dims = embed_dims
        self.head_dim = self.embed_dims // num_heads
        self.W_q = nn.Linear(embed_dims,embed_dims)
        self.W_k = nn.Linear(embed_dims,embed_dims)
        self.W_v = nn.Linear(embed_dims,embed_dims)
        self.W_o = nn.Linear(embed_dims,embed_dims)
    
    def embeddings(self,inp):
                max_val = 10
                embed_table = torch.randn(max_val, self.embed_dims)
                inp_seq = inp["Input Sequence"].tolist()
                embeds = [torch.stack([embed_table[x -1] for x in y]) for y in inp_seq]
                embeds = torch.stack(embeds)
                return embeds,embeds,embeds
    
    def split_heads(self,X):
            batch_size, seq_len, _ = X.shape
            X = X.view(batch_size, seq_len, self.num_head, self.head_dim)
            X = X.permute(0,2,1,3)
            return X
            
    def SDPA(self,Q,K,V):
        kd = torch.tensor(K.shape[-1],dtype=torch.float32)
        con_vecs = torch.matmul((F.softmax(torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(kd), dim=-1)),V)    
        return con_vecs

    def concat_heads(self,X):
        batch_size, num_heads, seq_len, head_dim = X.shape
        X = X.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return X
    
    def forward(self,inp):
        #Embeddings
        Q,K,V = self.embeddings(inp)
        #Linear Projections of Q,K,V
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        #Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        #Perfrom SDPA
        con_vecs = self.SDPA(Q,K,V)
        #concat heads
        con_vec = self.concat_heads(con_vecs)
        #Final Layer Linear Projection
        out = self.W_o(con_vec)

        return out

def generate_sequence_dataset(num_samples=100, sequence_length=5):
    # Generate random sequences of integers between 1 and 9
    input_sequences = [np.random.randint(1, 10, sequence_length).tolist() for _ in range(num_samples)]
    # The target sequences are simply the reversed version of the input sequences
    target_sequences = [seq[::-1] for seq in input_sequences]
    return input_sequences, target_sequences

# Generate a dataset with 100 samples of sequences of length 5
input_sequences, target_sequences = generate_sequence_dataset()

# Create a DataFrame for better visualization
data = {
    "Input Sequence": input_sequences,
    "Target Sequence": target_sequences
}

dataset_df = pd.DataFrame(data)

Att = MHA(5,5)
ot = Att.forward(dataset_df)
print(ot)