# import os
# import random
# import string
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from collections import Counter

# # Reproducibility: fix all random seeds for consistent results
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# # TASK 0: DATASET — Load 1000 Indian Names

# print("TASK 0: DATASET LOADING")

# names_path = os.path.join(OUTPUT_DIR, "TrainingNames.txt")

# # Read names from the TrainingNames.txt file
# if not os.path.exists(names_path):
#     raise FileNotFoundError(f"Could not find {names_path}. Please make sure 'TrainingNames.txt' exists in the script directory.")

# with open(names_path, 'r', encoding='utf-8') as f:
#     # Read lines, strip whitespace/newlines, and ignore empty lines
#     INDIAN_NAMES = [line.strip() for line in f if line.strip()]

# # Ensure we have exactly 1000 names (trim if needed)
# if len(INDIAN_NAMES) > 1000:
#     INDIAN_NAMES = INDIAN_NAMES[:1000]

# print(f"  Total Indian names loaded from dataset: {len(INDIAN_NAMES)}")
# print(f"  Loaded from: {names_path}")


# # Character-level encoding utilities

# # Build a character vocabulary from all names in the dataset
# # Special tokens: SOS (Start Of Sequence) and EOS (End Of Sequence) are
# # needed for autoregressive generation — the model starts with SOS and
# # generates characters until it produces EOS
# all_chars = sorted(set(''.join(INDIAN_NAMES).lower()))
# SOS_TOKEN = '<SOS>'
# EOS_TOKEN = '<EOS>'
# char_to_idx = {SOS_TOKEN: 0, EOS_TOKEN: 1}
# for i, ch in enumerate(all_chars):
#     char_to_idx[ch] = i + 2
# idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

# CHAR_VOCAB_SIZE = len(char_to_idx)
# SOS_IDX = char_to_idx[SOS_TOKEN]
# EOS_IDX = char_to_idx[EOS_TOKEN]

# print(f"  Character vocabulary size: {CHAR_VOCAB_SIZE}")
# print(f"  Characters: {all_chars}")


# def encode_name(name):
    
#     return [SOS_IDX] + [char_to_idx.get(ch, SOS_IDX) for ch in name.lower()] + [EOS_IDX]


# def decode_indices(indices):
    
#     result = []
#     for idx in indices:
#         if idx == EOS_IDX:
#             break
#         if idx == SOS_IDX:
#             continue
#         result.append(idx_to_char.get(idx, '?'))
#     return ''.join(result)


# # Prepare training data: each name becomes an encoded sequence
# training_sequences = [encode_name(name) for name in INDIAN_NAMES]

# # Determine the maximum sequence length for padding purposes
# MAX_SEQ_LEN = max(len(seq) for seq in training_sequences)
# print(f"  Maximum name length (with SOS/EOS): {MAX_SEQ_LEN}")

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"  Using device: {device}")


# # TASK 1: MODEL IMPLEMENTATION

# print("TASK 1: MODEL IMPLEMENTATION")

# # Shared hyperparameters for all models
# EMBEDDING_DIM = 64      # Dimension of character embeddings
# HIDDEN_SIZE = 128       # Hidden state size for RNN cells
# NUM_LAYERS = 1          # Number of stacked RNN layers
# LEARNING_RATE = 0.003   # Learning rate for Adam optimizer
# NUM_EPOCHS = 50         # Number of training passes over the dataset
# BATCH_SIZE = 64         # Mini-batch size for training




# class VanillaRNN(nn.Module):
    

#     def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
#         super(VanillaRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         # Character embedding layer: maps each character index to a dense vector
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)

#         # The RNN cell: processes sequences one step at a time, maintaining
#         # a hidden state that captures information from previous time steps
#         self.rnn = nn.RNN(
#             input_size=embedding_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True  # Input shape: (batch, seq_len, features)
#         )

#         # Output projection: maps hidden state to character probabilities
#         self.fc = nn.Linear(hidden_size, vocab_size)

#     def forward(self, x, hidden=None):
        
#         batch_size = x.size(0)
#         if hidden is None:
#             # Initialize hidden state with zeros for the start of a sequence
#             hidden = torch.zeros(self.num_layers, batch_size,
#                                  self.hidden_size, device=x.device)

#         # Embed characters: (batch, seq_len) -> (batch, seq_len, embed_dim)
#         embedded = self.embedding(x)

#         # Pass through RNN: output contains hidden states for all time steps
#         # output: (batch, seq_len, hidden_size)
#         # hidden: (num_layers, batch, hidden_size)
#         output, hidden = self.rnn(embedded, hidden)

#         # Project to vocabulary size for character prediction
#         output = self.fc(output)
#         return output, hidden

#     def generate(self, max_len=20, temperature=0.8):
        
#         self.eval()
#         with torch.no_grad():
#             # Start with the SOS token
#             current_char = torch.tensor([[SOS_IDX]], device=device)
#             hidden = None
#             generated = []

#             for _ in range(max_len):
#                 output, hidden = self(current_char, hidden)
#                 # Apply temperature scaling to the logits
#                 # Temperature < 1 makes the distribution sharper (more confident)
#                 # Temperature > 1 makes it flatter (more random)
#                 logits = output[0, -1] / temperature
#                 probs = torch.softmax(logits, dim=0)
#                 # Sample the next character from the probability distribution
#                 next_char_idx = torch.multinomial(probs, 1).item()

#                 if next_char_idx == EOS_IDX:
#                     break  # Stop generating when EOS is produced

#                 generated.append(next_char_idx)
#                 current_char = torch.tensor([[next_char_idx]], device=device)

#         return decode_indices(generated)


# # Model 2: Bidirectional LSTM (BLSTM)
# # Architecture: Embedding → Bidirectional LSTM → Linear → Softmax
# # BLSTM processes the sequence in both forward and backward directions,
# # capturing context from both past and future characters.
# # For generation, we use the concatenated forward and backward hidden states.

# class BiLSTMGenerator(nn.Module):
    

#     def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
#         super(BiLSTMGenerator, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.vocab_size = vocab_size

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)

#         # Bidirectional LSTM: output size is 2 * hidden_size because
#         # we concatenate forward and backward hidden states
#         self.lstm = nn.LSTM(
#             input_size=embedding_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True  # Key: processes in both directions
#         )

#         # Primary output layer: uses full bidirectional output (2*hidden)
#         self.fc = nn.Linear(hidden_size * 2, vocab_size)

#         # Forward-only prediction head: trained alongside the primary head
#         # using only the forward direction output. This head is used
#         # during generation when we can only look left-to-right.
#         self.fc_gen = nn.Linear(hidden_size, vocab_size)

#     def forward(self, x, hidden=None):
        
#         embedded = self.embedding(x)

#         # BLSTM combines forward and backward processing
#         # lstm_out shape: (batch, seq_len, 2 * hidden_size)
#         lstm_out, hidden = self.lstm(embedded, hidden)

#         # Main bidirectional projection
#         output = self.fc(lstm_out)

#         # Extract forward-only hidden states (first half of bidir output)
#         # and project through the generation head
#         fwd_hidden = lstm_out[:, :, :self.hidden_size]
#         fwd_output = self.fc_gen(fwd_hidden)

#         return output, hidden, fwd_output

#     def generate(self, max_len=20, temperature=0.8):
        
#         self.eval()
#         with torch.no_grad():
#             current_char = torch.tensor([[SOS_IDX]], device=device)
#             # Initialize hidden states for both directions
#             h = torch.zeros(self.num_layers * 2, 1, self.hidden_size,
#                             device=device)
#             c = torch.zeros(self.num_layers * 2, 1, self.hidden_size,
#                             device=device)
#             generated = []

#             for _ in range(max_len):
#                 embedded = self.embedding(current_char)

#                 # Run through bidirectional LSTM
#                 output, (h, c) = self.lstm(embedded, (h, c))

#                 # Extract only forward direction output (first half)
#                 fwd_output = output[:, :, :self.hidden_size]

#                 logits = self.fc_gen(fwd_output[0, -1]) / temperature
#                 probs = torch.softmax(logits, dim=0)
#                 next_char_idx = torch.multinomial(probs, 1).item()

#                 if next_char_idx == EOS_IDX:
#                     break
#                 generated.append(next_char_idx)
#                 current_char = torch.tensor([[next_char_idx]], device=device)

#         return decode_indices(generated)


# # Model 3: RNN with Basic Attention Mechanism
# # Architecture: Embedding → GRU → Attention → Linear → Softmax
# # The attention mechanism allows the model to selectively focus on
# # different parts of the generated sequence so far, rather than
# # relying solely on the most recent hidden state. This is especially
# # useful for longer names where the initial characters influence later ones.

# class RNNAttention(nn.Module):
    

#     def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
#         super(RNNAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)

#         # Use GRU as the recurrent cell (simpler than LSTM, works well in practice)
#         self.gru = nn.GRU(
#             input_size=embedding_dim,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True
#         )

#         # Attention layer: computes attention scores between current hidden
#         # state and all previous hidden states
#         # W_attn maps hidden_size → hidden_size for computing attention energy
#         self.attn_linear = nn.Linear(hidden_size, hidden_size)

#         # Combine attention context (hidden_size) with current hidden state
#         # (hidden_size) to produce the final output representation
#         self.combine = nn.Linear(hidden_size * 2, hidden_size)

#         # Project to vocabulary size for character prediction
#         self.fc = nn.Linear(hidden_size, vocab_size)

#     def forward(self, x, hidden=None):
        
#         batch_size = x.size(0)
#         seq_len = x.size(1)

#         if hidden is None:
#             hidden = torch.zeros(self.num_layers, batch_size,
#                                  self.hidden_size, device=x.device)

#         embedded = self.embedding(x)

#         # Process through GRU to get all hidden states
#         # all_hidden: (batch, seq_len, hidden_size) — hidden state at each step
#         all_hidden, hidden = self.gru(embedded, hidden)

#         # Apply self-attention: each position attends to all positions
#         # Step 1: Transform hidden states through attention linear layer
#         # keys: (batch, seq_len, hidden_size)
#         keys = self.attn_linear(all_hidden)

#         # Step 2: Compute attention scores using dot product
#         # We use causal masking so each position can only attend to
#         # previous positions (important for autoregressive generation)
#         # scores: (batch, seq_len, seq_len)
#         scores = torch.bmm(all_hidden, keys.transpose(1, 2))

#         # Apply causal mask: set future positions to -inf before softmax
#         # This prevents information leakage from future characters
#         causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device),
#                                 diagonal=1).bool()
#         scores.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))

#         # Normalize attention weights
#         attn_weights = torch.softmax(scores, dim=-1)

#         # Step 3: Compute attention context as weighted sum of hidden states
#         # context: (batch, seq_len, hidden_size)
#         context = torch.bmm(attn_weights, all_hidden)

#         # Step 4: Combine attention context with current hidden state
#         combined = torch.cat([all_hidden, context], dim=-1)
#         combined = torch.tanh(self.combine(combined))

#         # Project to vocabulary
#         output = self.fc(combined)
#         return output, hidden

#     def generate(self, max_len=20, temperature=0.8):
        
#         self.eval()
#         with torch.no_grad():
#             current_char = torch.tensor([[SOS_IDX]], device=device)
#             hidden = None
#             generated = []
#             hidden_states = []

#             for step in range(max_len):
#                 embedded = self.embedding(current_char)
#                 gru_output, hidden = self.gru(embedded, hidden)
#                 hidden_states.append(gru_output.squeeze(1))

#                 # Stack all hidden states collected so far
#                 all_h = torch.stack(hidden_states, dim=1)  # (1, step+1, hidden)

#                 # Compute attention over previous hidden states
#                 keys = self.attn_linear(all_h)
#                 current_h = gru_output  # (1, 1, hidden)
#                 # Attention scores: dot product of current state with all keys
#                 scores = torch.bmm(current_h, keys.transpose(1, 2))
#                 attn_weights = torch.softmax(scores, dim=-1)
#                 context = torch.bmm(attn_weights, all_h)

#                 # Combine current hidden state with attention context
#                 combined = torch.cat([gru_output, context], dim=-1)
#                 combined = torch.tanh(self.combine(combined))

#                 logits = self.fc(combined[0, -1]) / temperature
#                 probs = torch.softmax(logits, dim=0)
#                 next_char_idx = torch.multinomial(probs, 1).item()

#                 if next_char_idx == EOS_IDX:
#                     break
#                 generated.append(next_char_idx)
#                 current_char = torch.tensor([[next_char_idx]], device=device)

#         return decode_indices(generated)


# # Instantiate all three models and report their architectures

# models = {
#     'VanillaRNN': VanillaRNN(CHAR_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE,
#                               NUM_LAYERS).to(device),
#     'BiLSTM': BiLSTMGenerator(CHAR_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE,
#                                 NUM_LAYERS).to(device),
#     'RNNAttention': RNNAttention(CHAR_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE,
#                                   NUM_LAYERS).to(device),
# }


# def count_parameters(model):
#     """Counts the number of trainable parameters in a PyTorch model."""
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print("\n  Model Architectures and Parameter Counts:")
# print("  " + "-" * 60)
# for name, model in models.items():
#     param_count = count_parameters(model)
#     print(f"\n  {name}:")
#     print(f"    Architecture: {model.__class__.__name__}")
#     print(f"    Embedding dim: {EMBEDDING_DIM}")
#     print(f"    Hidden size:   {HIDDEN_SIZE}")
#     print(f"    Num layers:    {NUM_LAYERS}")
#     print(f"    Trainable parameters: {param_count:,}")

# print("\n  Hyperparameters:")
# print(f"    Learning rate: {LEARNING_RATE}")
# print(f"    Batch size:    {BATCH_SIZE}")
# print(f"    Epochs:        {NUM_EPOCHS}")


# # Training utility

# def pad_sequences(sequences, pad_value=0):
    
#     max_len = max(len(s) for s in sequences)
#     padded = [s + [pad_value] * (max_len - len(s)) for s in sequences]
#     return torch.tensor(padded, dtype=torch.long)


# def train_model(model, training_sequences, model_name, num_epochs=NUM_EPOCHS,
#                 lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    
#     print(f"\n  Training {model_name}...")

#     criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     loss_history = []
#     model.train()

#     for epoch in range(num_epochs):
#         # Shuffle training data each epoch for better generalization
#         indices = list(range(len(training_sequences)))
#         random.shuffle(indices)

#         total_loss = 0.0
#         n_batches = 0

#         for start in range(0, len(indices), batch_size):
#             batch_indices = indices[start:start + batch_size]
#             batch_seqs = [training_sequences[i] for i in batch_indices]

#             # Pad sequences to the same length within this batch
#             padded = pad_sequences(batch_seqs).to(device)

#             # Input: all characters except the last (teacher forcing)
#             # Target: all characters shifted by one position
#             input_seq = padded[:, :-1]
#             target_seq = padded[:, 1:]

#             # Forward pass — BiLSTM returns 3 values (output, hidden, fwd_output)
#             # while other models return 2 values (output, hidden)
#             result = model(input_seq)
#             if len(result) == 3:
#                 # BiLSTM: train both bidirectional and forward-only heads
#                 output, _, fwd_output = result
#                 output = output.reshape(-1, CHAR_VOCAB_SIZE)
#                 fwd_output = fwd_output.reshape(-1, CHAR_VOCAB_SIZE)
#                 target = target_seq.reshape(-1)
#                 # Combined loss: bidirectional + forward-only prediction
#                 # The forward-only loss ensures fc_gen learns for generation
#                 loss = criterion(output, target) + criterion(fwd_output, target)
#             else:
#                 output, _ = result
#                 # Reshape for cross-entropy: (batch*seq_len, vocab) vs (batch*seq_len,)
#                 output = output.reshape(-1, CHAR_VOCAB_SIZE)
#                 target = target_seq.reshape(-1)
#                 loss = criterion(output, target)

#             # Backward pass and weight update
#             optimizer.zero_grad()
#             loss.backward()
#             # Gradient clipping prevents exploding gradients, which is
#             # a common problem with RNNs on longer sequences
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()

#             total_loss += loss.item()
#             n_batches += 1

#         avg_loss = total_loss / max(n_batches, 1)
#         loss_history.append(avg_loss)

#         # Print progress every 10 epochs
#         if (epoch + 1) % 10 == 0:
#             print(f"    Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

#     return loss_history


# # Train all three models
# print("\n  Training all models (this may take a few minutes)...")
# loss_histories = {}
# for name, model in models.items():
#     loss_histories[name] = train_model(model, training_sequences, name)


# # TASK 2: QUANTITATIVE EVALUATION

# print("TASK 2: QUANTITATIVE EVALUATION")

# # Convert training names to a set for fast lookup when computing novelty
# training_set = set(name.lower() for name in INDIAN_NAMES)
# NUM_GENERATED = 100  # Generate 100 names per model for evaluation


# def evaluate_model(model, model_name, num_samples=NUM_GENERATED):
    
#     generated_names = []
#     for _ in range(num_samples):
#         name = model.generate(max_len=20, temperature=0.8)
#         generated_names.append(name)

#     # Novelty: how many generated names are NOT in the training set
#     novel_count = sum(1 for name in generated_names
#                       if name.lower() not in training_set)
#     novelty_rate = novel_count / len(generated_names) * 100

#     # Diversity: fraction of generated names that are unique
#     unique_names = set(generated_names)
#     diversity = len(unique_names) / len(generated_names) * 100

#     print(f"\n  {model_name}:")
#     print(f"    Generated: {len(generated_names)} names")
#     print(f"    Novelty Rate: {novelty_rate:.1f}% "
#           f"({novel_count}/{len(generated_names)} not in training set)")
#     print(f"    Diversity: {diversity:.1f}% "
#           f"({len(unique_names)}/{len(generated_names)} unique)")

#     return {
#         'names': generated_names,
#         'novelty_rate': novelty_rate,
#         'diversity': diversity,
#     }


# # Evaluate each model
# evaluation_results = {}
# for name, model in models.items():
#     evaluation_results[name] = evaluate_model(model, name)

# # Print comparison table
# print("\n  " + "-" * 50)
# print(f"  {'Model':<20} {'Novelty %':<15} {'Diversity %':<15}")
# print("  " + "-" * 50)
# for name in models:
#     r = evaluation_results[name]
#     print(f"  {name:<20} {r['novelty_rate']:<15.1f} {r['diversity']:<15.1f}")
# print("  " + "-" * 50)


# # TASK 3: QUALITATIVE ANALYSIS

# print("TASK 3: QUALITATIVE ANALYSIS")

# for model_name in models:
#     result = evaluation_results[model_name]
#     names = result['names']

#     print(f"\n  {model_name} — Representative Generated Samples:")
#     print(f"    {', '.join(names[:20])}")

#     # Analyze failure modes
#     short_names = [n for n in names if len(n) <= 1]
#     long_names = [n for n in names if len(n) > 15]
#     repeated_chars = [n for n in names if any(
#         n.count(c) > len(n) * 0.5 for c in set(n) if c.isalpha()
#     )]

#     print(f"\n    Failure Analysis:")
#     print(f"    - Too short (≤1 char): {len(short_names)}")
#     if short_names:
#         print(f"      Examples: {short_names[:5]}")
#     print(f"    - Too long (>15 chars): {len(long_names)}")
#     if long_names:
#         print(f"      Examples: {long_names[:5]}")
#     print(f"    - Repetitive characters: {len(repeated_chars)}")
#     if repeated_chars:
#         print(f"      Examples: {repeated_chars[:5]}")

# # Discussion
# print("\n  --- Discussion ---")
# print("  Vanilla RNN: Simple architecture, tends to produce shorter names.")
# print("  It may struggle with longer dependencies but generates reasonable")
# print("  character sequences. Common failure: repeating patterns or abrupt stops.")
# print()
# print("  BiLSTM: Better at capturing both forward and backward context during")
# print("  training, but generation is still left-to-right. Often produces more")
# print("  diverse and natural-sounding names due to richer learned representations.")
# print()
# print("  RNN with Attention: The attention mechanism helps maintain consistency")
# print("  across longer names by allowing the model to revisit earlier characters.")
# print("  Typically produces the most realistic names but may occasionally")
# print("  generate unusual character combinations when attention weights are noisy.")


# # Print Summary

# # Save the trained models for later evaluation
# print("\n  Saving models to disk...")
# torch.save(models['VanillaRNN'].state_dict(), os.path.join(OUTPUT_DIR, "vanilla_rnn.pkl"))
# torch.save(models['BiLSTM'].state_dict(), os.path.join(OUTPUT_DIR, "bilstm.pkl"))
# torch.save(models['RNNAttention'].state_dict(), os.path.join(OUTPUT_DIR, "rnn_attention.pkl"))
# import json
# with open(os.path.join(OUTPUT_DIR, "char_vocab.json"), "w") as f:
#     json.dump(char_to_idx, f)
# print("  Models and vocabulary saved.")

# print("\n" + "=" * 70)
# print("ALL TASKS COMPLETED SUCCESSFULLY")
# print("=" * 70)
# print(f"\nOutput files:")
# print(f"  - {names_path}")
# print(f"\nModel Details:")
# for name, model in models.items():
#     print(f"  - {name}: {count_parameters(model):,} trainable parameters")


import os
import random
import string
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

# Reproducibility: fix all random seeds for consistent results
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# TASK 0: DATASET — Load 1000 Indian Names

print("TASK 0: DATASET LOADING")

names_path = os.path.join(OUTPUT_DIR, "TrainingNames.txt")

# Read names from the TrainingNames.txt file
if not os.path.exists(names_path):
    raise FileNotFoundError(f"Could not find {names_path}. Please make sure 'TrainingNames.txt' exists in the script directory.")

with open(names_path, 'r', encoding='utf-8') as f:
    # Read lines, strip whitespace/newlines, and ignore empty lines
    INDIAN_NAMES = [line.strip() for line in f if line.strip()]

# Ensure we have exactly 1000 names (trim if needed)
if len(INDIAN_NAMES) > 1000:
    INDIAN_NAMES = INDIAN_NAMES[:1000]

print(f"  Total Indian names loaded from dataset: {len(INDIAN_NAMES)}")
print(f"  Loaded from: {names_path}")


# Character-level encoding utilities
all_chars = sorted(set(''.join(INDIAN_NAMES).lower()))
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
char_to_idx = {SOS_TOKEN: 0, EOS_TOKEN: 1}
for i, ch in enumerate(all_chars):
    char_to_idx[ch] = i + 2
idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

CHAR_VOCAB_SIZE = len(char_to_idx)
SOS_IDX = char_to_idx[SOS_TOKEN]
EOS_IDX = char_to_idx[EOS_TOKEN]

print(f"  Character vocabulary size: {CHAR_VOCAB_SIZE}")

def encode_name(name):
    return [SOS_IDX] + [char_to_idx.get(ch, SOS_IDX) for ch in name.lower()] + [EOS_IDX]

def decode_indices(indices):
    result = []
    for idx in indices:
        if idx == EOS_IDX:
            break
        if idx == SOS_IDX:
            continue
        result.append(idx_to_char.get(idx, '?'))
    return ''.join(result)


# Prepare training data: each name becomes an encoded sequence
training_sequences = [encode_name(name) for name in INDIAN_NAMES]

# Determine the maximum sequence length for padding purposes
MAX_SEQ_LEN = max(len(seq) for seq in training_sequences)
print(f"  Maximum name length (with SOS/EOS): {MAX_SEQ_LEN}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Using device: {device}")


# TASK 1: MODEL IMPLEMENTATION

print("\nTASK 1: MODEL IMPLEMENTATION")

# Shared hyperparameters for all models
EMBEDDING_DIM = 64      
HIDDEN_SIZE = 128       
NUM_LAYERS = 1          
LEARNING_RATE = 0.003   
NUM_EPOCHS = 50         
BATCH_SIZE = 64         

# ==========================================
# Model 1: Vanilla RNN
# ==========================================
class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size,
                                 self.hidden_size, device=x.device)

        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def generate(self, max_len=20, temperature=0.8):
        self.eval()
        with torch.no_grad():
            current_char = torch.tensor([[SOS_IDX]], device=device)
            hidden = None
            generated = []

            for _ in range(max_len):
                output, hidden = self(current_char, hidden)
                logits = output[0, -1] / temperature
                probs = torch.softmax(logits, dim=0)
                next_char_idx = torch.multinomial(probs, 1).item()

                if next_char_idx == EOS_IDX:
                    break
                generated.append(next_char_idx)
                current_char = torch.tensor([[next_char_idx]], device=device)

        return decode_indices(generated)


# ==========================================
# Model 2: Bidirectional LSTM (Dual-Head)
# ==========================================
class BiLSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(BiLSTMGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  
        )

        # Primary output layer (bidirectional)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

        # Forward-only prediction head for generation
        self.fc_gen = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Main bidirectional projection
        output = self.fc(lstm_out)

        # Extract forward-only hidden states and project through the generation head
        fwd_hidden = lstm_out[:, :, :self.hidden_size]
        fwd_output = self.fc_gen(fwd_hidden)

        return output, hidden, fwd_output

    def generate(self, max_len=20, temperature=0.8):
        self.eval()
        with torch.no_grad():
            current_char = torch.tensor([[SOS_IDX]], device=device)
            h = torch.zeros(self.num_layers * 2, 1, self.hidden_size, device=device)
            c = torch.zeros(self.num_layers * 2, 1, self.hidden_size, device=device)
            generated = []

            for _ in range(max_len):
                embedded = self.embedding(current_char)
                output, (h, c) = self.lstm(embedded, (h, c))
                
                # Extract only forward direction output
                fwd_output = output[:, :, :self.hidden_size]
                logits = self.fc_gen(fwd_output[0, -1]) / temperature
                probs = torch.softmax(logits, dim=0)
                next_char_idx = torch.multinomial(probs, 1).item()

                if next_char_idx == EOS_IDX:
                    break
                generated.append(next_char_idx)
                current_char = torch.tensor([[next_char_idx]], device=device)

        return decode_indices(generated)

class CharBLSTM(nn.Module):
    """
    Character-Level Bidirectional LSTM for Name Generation.

    Architecture:
        - Embedding layer: character indices → dense vectors
        - Forward LSTM: processes sequence left-to-right
        - Backward LSTM: processes sequence right-to-left
        - Concatenation: forward + backward hidden states
        - Linear output: 2*hidden_size → vocab_size

    Note: For generation (autoregressive), we primarily use the forward direction.
    The backward pass helps during training with teacher forcing.

    Hyperparameters:
        vocab_size: Number of unique characters
        embedding_dim: Character embedding dimension (default: 32)
        hidden_size: Hidden state size per direction (default: 128)
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=128, dropout=0.1):
        super(CharBLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Forward and backward LSTM cells (from scratch)
        self.forward_cell = LSTMCell(embedding_dim, hidden_size)
        self.backward_cell = LSTMCell(embedding_dim, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer: concatenated bidirectional hidden → vocab logits
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass (bidirectional).

        Args:
            x: (batch_size, seq_len) — character indices
            hidden: ignored (initialized internally for each direction)

        Returns:
            output: (batch_size, seq_len, vocab_size) — logits
            hidden: tuple of final (h_fwd, c_fwd, h_bwd, c_bwd)
        """
        batch_size, seq_len = x.size()
        device = x.device

        # Embed
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # Initialize states
        h_fwd = torch.zeros(batch_size, self.hidden_size, device=device)
        c_fwd = torch.zeros(batch_size, self.hidden_size, device=device)
        h_bwd = torch.zeros(batch_size, self.hidden_size, device=device)
        c_bwd = torch.zeros(batch_size, self.hidden_size, device=device)

        # Forward pass: left to right
        fwd_outputs = []
        for t in range(seq_len):
            h_fwd, c_fwd = self.forward_cell(emb[:, t, :], h_fwd, c_fwd)
            fwd_outputs.append(h_fwd)

        # Backward pass: right to left
        bwd_outputs = []
        for t in range(seq_len - 1, -1, -1):
            h_bwd, c_bwd = self.backward_cell(emb[:, t, :], h_bwd, c_bwd)
            bwd_outputs.insert(0, h_bwd)

        # Stack and concatenate
        fwd_out = torch.stack(fwd_outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        bwd_out = torch.stack(bwd_outputs, dim=1)
        combined = torch.cat([fwd_out, bwd_out], dim=2)  # (batch_size, seq_len, 2*hidden_size)
        combined = self.dropout(combined)

        # Project to vocab
        logits = self.fc_out(combined)
        return logits, (h_fwd, c_fwd, h_bwd, c_bwd)

    def generate_forward_only(self, x, hidden=None):
        """
        Forward-only pass for autoregressive generation.
        During generation, we can only use the forward direction.

        Args:
            x: (batch_size, seq_len) — input characters
            hidden: tuple (h, c) or None

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            hidden: (h, c) — final forward hidden state
        """
        batch_size, seq_len = x.size()
        device = x.device

        emb = self.embedding(x)

        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            c = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h, c = hidden

        outputs = []
        for t in range(seq_len):
            h, c = self.forward_cell(emb[:, t, :], h, c)
            outputs.append(h)

        out = torch.stack(outputs, dim=1)

        # Use only forward hidden size — we need to project through a separate layer
        # For generation, use a simpler linear projection
        # We reuse half of fc_out weights conceptually, but for simplicity use a dedicated layer
        logits = self.fc_out(torch.cat([out, torch.zeros_like(out)], dim=2))
        return logits, (h, c)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))


# ==========================================
# Model 3: RNN with Attention
# ==========================================
class RNNAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(RNNAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.attn_linear = nn.Linear(hidden_size, hidden_size)
        self.combine = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size,
                                 self.hidden_size, device=x.device)

        embedded = self.embedding(x)
        all_hidden, hidden = self.gru(embedded, hidden)

        keys = self.attn_linear(all_hidden)
        scores = torch.bmm(all_hidden, keys.transpose(1, 2))

        # Causal mask to prevent looking ahead
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device),
                                diagonal=1).bool()
        scores.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, all_hidden)

        combined = torch.cat([all_hidden, context], dim=-1)
        combined = torch.tanh(self.combine(combined))

        output = self.fc(combined)
        return output, hidden

    def generate(self, max_len=20, temperature=0.8):
        self.eval()
        with torch.no_grad():
            current_char = torch.tensor([[SOS_IDX]], device=device)
            hidden = None
            generated = []
            hidden_states = []

            for step in range(max_len):
                embedded = self.embedding(current_char)
                gru_output, hidden = self.gru(embedded, hidden)
                hidden_states.append(gru_output.squeeze(1))

                all_h = torch.stack(hidden_states, dim=1) 
                keys = self.attn_linear(all_h)
                current_h = gru_output  
                
                scores = torch.bmm(current_h, keys.transpose(1, 2))
                attn_weights = torch.softmax(scores, dim=-1)
                context = torch.bmm(attn_weights, all_h)

                combined = torch.cat([gru_output, context], dim=-1)
                combined = torch.tanh(self.combine(combined))

                logits = self.fc(combined[0, -1]) / temperature
                probs = torch.softmax(logits, dim=0)
                next_char_idx = torch.multinomial(probs, 1).item()

                if next_char_idx == EOS_IDX:
                    break
                generated.append(next_char_idx)
                current_char = torch.tensor([[next_char_idx]], device=device)

        return decode_indices(generated)


# Instantiate models 
models = {
    'VanillaRNN': VanillaRNN(CHAR_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device),
    'BiLSTM': BiLSTMGenerator(CHAR_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device),
    'RNNAttention': RNNAttention(CHAR_VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device),
}

def count_parameters(model):
    """Counts the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n  Model Architectures and Parameter Counts:")
print("  " + "-" * 60)
for name, model in models.items():
    param_count = count_parameters(model)
    print(f"\n  {name}:")
    print(f"    Architecture: {model.__class__.__name__}")
    print(f"    Trainable parameters: {param_count:,}")

print("\n  Hyperparameters:")
print(f"    Learning rate: {LEARNING_RATE}")
print(f"    Batch size:    {BATCH_SIZE}")
print(f"    Epochs:        {NUM_EPOCHS}")


# Training utility
def pad_sequences(sequences, pad_value=0):
    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_value] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)

def train_model(model, training_sequences, model_name, num_epochs=NUM_EPOCHS,
                lr=LEARNING_RATE, batch_size=BATCH_SIZE):
    
    print(f"\n  Training {model_name}...")
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    model.train()

    for epoch in range(num_epochs):
        indices = list(range(len(training_sequences)))
        random.shuffle(indices)

        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_seqs = [training_sequences[i] for i in batch_indices]

            padded = pad_sequences(batch_seqs).to(device)

            # Teacher forcing
            input_seq = padded[:, :-1]
            target_seq = padded[:, 1:]

            # Forward pass: handle varying return sizes dynamically
            result = model(input_seq)
            
            if len(result) == 3:
                # BiLSTM returns 3 items: output, hidden, fwd_output
                output, _, fwd_output = result
                output = output.reshape(-1, CHAR_VOCAB_SIZE)
                fwd_output = fwd_output.reshape(-1, CHAR_VOCAB_SIZE)
                target = target_seq.reshape(-1)
                
                # Dual-loss calculation for the BiLSTM workaround
                loss = criterion(output, target) + criterion(fwd_output, target)
            else:
                # Other models return 2 items: output, hidden
                output, _ = result
                output = output.reshape(-1, CHAR_VOCAB_SIZE)
                target = target_seq.reshape(-1)
                loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return loss_history


# Train models
print("\n  Training all models (this may take a few minutes)...")
loss_histories = {}
for name, model in models.items():
    loss_histories[name] = train_model(model, training_sequences, name)


# TASK 2: QUANTITATIVE EVALUATION
print("\nTASK 2: QUANTITATIVE EVALUATION")

training_set = set(name.lower() for name in INDIAN_NAMES)
NUM_GENERATED = 100 

def evaluate_model(model, model_name, num_samples=NUM_GENERATED):
    generated_names = []
    for _ in range(num_samples):
        name = model.generate(max_len=20, temperature=0.8)
        generated_names.append(name)

    novel_count = sum(1 for name in generated_names if name.lower() not in training_set)
    novelty_rate = novel_count / len(generated_names) * 100

    unique_names = set(generated_names)
    diversity = len(unique_names) / len(generated_names) * 100

    return {
        'names': generated_names,
        'novelty_rate': novelty_rate,
        'diversity': diversity,
    }

evaluation_results = {}
for name, model in models.items():
    evaluation_results[name] = evaluate_model(model, name)

print("\n  " + "-" * 50)
print(f"  {'Model':<20} {'Novelty %':<15} {'Diversity %':<15}")
print("  " + "-" * 50)
for name in models:
    r = evaluation_results[name]
    print(f"  {name:<20} {r['novelty_rate']:<15.1f} {r['diversity']:<15.1f}")
print("  " + "-" * 50)


# TASK 3: QUALITATIVE ANALYSIS
print("\nTASK 3: QUALITATIVE ANALYSIS")

for model_name in models:
    result = evaluation_results[model_name]
    names = result['names']

    print(f"\n  {model_name} — Representative Generated Samples:")
    print(f"    {', '.join(names[:20])}")

    short_names = [n for n in names if len(n) <= 1]
    long_names = [n for n in names if len(n) > 15]
    repeated_chars = [n for n in names if any(n.count(c) > len(n) * 0.5 for c in set(n) if c.isalpha())]

    print(f"    Failure Analysis:")
    print(f"    - Too short (≤1 char): {len(short_names)}")
    print(f"    - Too long (>15 chars): {len(long_names)}")
    print(f"    - Repetitive characters: {len(repeated_chars)}")

print("\n  --- Discussion ---")
print("  Vanilla RNN: Simple architecture, tends to produce shorter names.")
print("  BiLSTM: Dual-head architecture trained simultaneously on bidirectional")
print("  and forward-only sequences; utilizes forward-only head for generation.")
print("  RNN with Attention: Often produces the most realistic names by ")
print("  referencing earlier characters dynamically via the causal attention mask.")


# Save Models
print("\n  Saving models to disk...")
torch.save(models['VanillaRNN'].state_dict(), os.path.join(OUTPUT_DIR, "vanilla_rnn.pkl"))
torch.save(models['BiLSTM'].state_dict(), os.path.join(OUTPUT_DIR, "bilstm.pkl"))
torch.save(models['RNNAttention'].state_dict(), os.path.join(OUTPUT_DIR, "rnn_attention.pkl"))

with open(os.path.join(OUTPUT_DIR, "char_vocab.json"), "w") as f:
    json.dump(char_to_idx, f)
print("  Models and vocabulary saved.")

print("\n" + "=" * 70)
print("ALL TASKS COMPLETED SUCCESSFULLY")
print("=" * 70)