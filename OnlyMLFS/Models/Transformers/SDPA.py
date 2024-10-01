'''This is the implmentation of Scaled Dot Product Attention \n
Input - Query Q, Value V and Key K of dimension dₖ \n
Attention(Q,K,V) = softmax(QK^T/sqroot(dₖ)) * V
'''
'''
Input and formatting
'''

import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
set_seed = 420
np.random.seed(set_seed)

# Generate a toy dataset for testing attention mechanism
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
#Embeds
embed_dims = 5
max_val = 10

embed_table = np.random.randn(max_val, embed_dims)
inp_seq = dataset_df["Input Sequence"].tolist()
embeds = [np.array([embed_table[x -1 ] for x in y]) for y in inp_seq]
#Attention Inputs
Q = embeds[-1]
K = np.array(embeds)
V = np.array(embeds)
#Apply!

scores = np.dot(K, Q)
tensor = torch.tensor(scores, dtype=torch.float32)
W = F.softmax(tensor,dim=-1)
W_np = W.numpy()
con_vec = np.sum(W_np[:, np.newaxis] * V, axis=0)
#print(con_vec)

def SDPA(Q,K,V):
    kd = (K.shape)[0]
    Q_t = torch.tensor(Q, dtype=torch.float32)
    Q_t = Q_t.unsqueeze(0)
    K_t = torch.tensor(K, dtype=torch.float32)
    V_t = torch.tensor(V, dtype=torch.float32)
    attention = (F.softmax(torch.matmul(Q_t, K_t) / np.sqrt(kd)) ) * V_t
    print(attention)
    return attention

