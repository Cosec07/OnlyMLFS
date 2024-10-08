import numpy as np
import pandas as pd
import OnlyMLFS.Models.Transformers.Encoder as OML
import torch
import torch.nn as nn
import torch.optim as optim

def generate_sequence_dataset(num_samples=100, sequence_length=5, vocab_size=10):
    """
    Generate a random sequence dataset of integers for training.
    Each sequence will consist of random integers between 1 and vocab_size.
    
    Args:
        num_samples (int): Number of samples in the dataset.
        sequence_length (int): Length of each sequence.
        vocab_size (int): Number of unique tokens in the vocabulary.
    
    Returns:
        pd.DataFrame: A DataFrame with Input Sequence and Target Sequence columns.
    """
    # Generate random sequences of integers between 1 and vocab_size
    input_sequences = [np.random.randint(0, vocab_size, sequence_length).tolist() for _ in range(num_samples)]
    # The target sequences can be the reversed version of the input sequences for simplicity
    target_sequences = [seq[::-1] for seq in input_sequences]
    
    # Create a DataFrame for better visualization
    data = {
        "Input Sequence": input_sequences,
        "Target Sequence": target_sequences
    }
    dataset_df = pd.DataFrame(data)
    return dataset_df

# Generate a toy dataset with 100 samples of sequences of length 5
num_samples = 1000
sequence_length = 5
vocab_size = 10
data = generate_sequence_dataset(num_samples, sequence_length, vocab_size)

# Convert the data into dataframe and test the component
num_layers = 5
num_heads = 6
embedding_dim = 6
network = OML.Encoder(num_layers, num_heads, embedding_dim)
criteria = nn.CrossEntropyLoss()
opt = optim.Adam(network.parameters(), lr=0.001)

#prepare data
inp = [torch.tensor(seq,dtype=torch.long) for seq in data['Input Sequence']]
targets = [torch.tensor(seq,dtype=torch.long) for seq in data['Target Sequence']]

#training loop
num_epochs = 10

for epochs in range(num_epochs):
    epoch_loss = 0
    for inp_seq,tar_seq in zip(inp,targets):
        opt.zero_grad()

        inp_seq = inp_seq.unsqueeze(0)
        out = network.forward(inp_seq)

        loss = criteria(out.view(-1,embedding_dim), tar_seq.view(-1))

        #backprop
        loss.backward()
        optim.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(inp)}")