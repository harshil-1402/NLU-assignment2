import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import json
import random

# Setup
OUTPUT_DIR = "/home/harshil/Documents/NLU/sol2-nlu"
EMBEDDING_DIM = 300
WINDOW_SIZE = 5
NUM_EPOCHS = 10  # 10 is enough for a quick demo
BATCH_SIZE = 128

# Load corpus
corpus_path = os.path.join(OUTPUT_DIR, "cleaned_corpus.txt")
with open(corpus_path, "r") as f:
    corpus_tokens = f.read().split()

# Vocab
word_counts = Counter(corpus_tokens)
vocab = {word: idx for idx, (word, count) in enumerate(
    sorted(word_counts.items(), key=lambda x: -x[1])
) if count >= 2}
vocab['<UNK>'] = len(vocab)
corpus_indices = [vocab.get(w, vocab['<UNK>']) for w in corpus_tokens]

# Model
class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNS, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
    def forward(self, center_idx, context_idx):
        return (self.center_embeddings(center_idx) * self.context_embeddings(context_idx)).sum(dim=1)

device = torch.device('cpu')
model = SkipGramNS(len(vocab), EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

# Negative Sampling Dist
freq = np.zeros(len(vocab))
for idx in corpus_indices: freq[idx] += 1
neg_dist = (freq ** 0.75) / (freq ** 0.75).sum()

# Dataset
dataset = []
for i in range(WINDOW_SIZE, len(corpus_indices) - WINDOW_SIZE):
    for j in range(i - WINDOW_SIZE, i + WINDOW_SIZE + 1):
        if j != i: dataset.append((corpus_indices[i], corpus_indices[j]))

# Train
print(f"Training 300D Skip-gram on {len(vocab)} words...")
for epoch in range(NUM_EPOCHS):
    random.shuffle(dataset)
    total_loss = 0
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i:i+BATCH_SIZE]
        if not batch: continue
        
        curr_bs = len(batch)
        centers = [p[0] for p in batch]
        contexts = [p[1] for p in batch]
        
        # Positives
        c_pos = torch.tensor(centers)
        ctx_pos = torch.tensor(contexts)
        lbl_pos = torch.ones(curr_bs)
        
        # Negatives (5 per positive)
        neg_samples = np.random.choice(len(vocab), size=(curr_bs, 5), p=neg_dist)
        c_neg = torch.tensor(centers).repeat_interleave(5)
        ctx_neg = torch.tensor(neg_samples.flatten())
        lbl_neg = torch.zeros(curr_bs * 5)
        
        # Combine
        c = torch.cat([c_pos, c_neg])
        ctx = torch.cat([ctx_pos, ctx_neg])
        lbl = torch.cat([lbl_pos, lbl_neg])
        
        optimizer.zero_grad()
        loss = criterion(model(c, ctx), lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/ (len(dataset)//BATCH_SIZE + 1):.4f}")

# Save
emb_file = os.path.join(OUTPUT_DIR, "sg_embeddings_300d.npy")
vocab_file = os.path.join(OUTPUT_DIR, "vocab_300d.json")
np.save(emb_file, model.center_embeddings.weight.detach().numpy())
with open(vocab_file, "w") as f:
    json.dump(vocab, f)
print(f"Saved 300D embeddings to {emb_file}")
print(f"Saved vocab to {vocab_file}")

# Pick a word and print its first 10 dims
target_word = "institute"
if target_word in vocab:
    idx = vocab[target_word]
    vec = model.center_embeddings.weight[idx].detach().numpy()
    print(f"\nWord: {target_word}")
    print(f"Embedding (first 10 of 300): {vec[:10]}")
