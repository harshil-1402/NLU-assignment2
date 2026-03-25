import os
import json
import torch
import torch.nn as nn
import numpy as np
import urllib.request  

# Use the same device as training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Model Definitions (Same as name_generation.py) ---

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(VanillaRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden

    def generate(self, char_to_idx, idx_to_char, max_len=20, temperature=0.8):
        self.eval()
        SOS_IDX = char_to_idx['<SOS>']
        EOS_IDX = char_to_idx['<EOS>']
        with torch.no_grad():
            current_char = torch.tensor([[SOS_IDX]], device=device)
            hidden = None
            generated = []
            for _ in range(max_len):
                output, hidden = self(current_char, hidden)
                logits = output[0, -1] / temperature
                probs = torch.softmax(logits, dim=0)
                next_char_idx = torch.multinomial(probs, 1).item()
                if next_char_idx == EOS_IDX: break
                generated.append(next_char_idx)
                current_char = torch.tensor([[next_char_idx]], device=device)
        return ''.join([idx_to_char[str(idx)] for idx in generated])

class BiLSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(BiLSTMGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.fc_gen = nn.Linear(hidden_size, vocab_size)

    def generate(self, char_to_idx, idx_to_char, max_len=20, temperature=0.8):
        self.eval()
        SOS_IDX = char_to_idx['<SOS>']
        EOS_IDX = char_to_idx['<EOS>']
        with torch.no_grad():
            current_char = torch.tensor([[SOS_IDX]], device=device)
            h = torch.zeros(self.num_layers * 2, 1, self.hidden_size, device=device)
            c = torch.zeros(self.num_layers * 2, 1, self.hidden_size, device=device)
            generated = []
            for _ in range(max_len):
                embedded = self.embedding(current_char)
                output, (h, c) = self.lstm(embedded, (h, c))
                fwd_output = output[:, :, :self.hidden_size]
                logits = self.fc_gen(fwd_output[0, -1]) / temperature
                probs = torch.softmax(logits, dim=0)
                next_char_idx = torch.multinomial(probs, 1).item()
                if next_char_idx == EOS_IDX: break
                generated.append(next_char_idx)
                current_char = torch.tensor([[next_char_idx]], device=device)
        return ''.join([idx_to_char[str(idx)] for idx in generated])

class RNNAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1):
        super(RNNAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.attn_linear = nn.Linear(hidden_size, hidden_size)
        self.combine = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def generate(self, char_to_idx, idx_to_char, max_len=20, temperature=0.8):
        self.eval()
        SOS_IDX = char_to_idx['<SOS>']
        EOS_IDX = char_to_idx['<EOS>']
        with torch.no_grad():
            current_char = torch.tensor([[SOS_IDX]], device=device)
            hidden = None
            generated = []
            hidden_states = []
            for _ in range(max_len):
                embedded = self.embedding(current_char)
                gru_output, hidden = self.gru(embedded, hidden)
                hidden_states.append(gru_output.squeeze(1))
                all_h = torch.stack(hidden_states, dim=1)
                keys = self.attn_linear(all_h)
                scores = torch.bmm(gru_output, keys.transpose(1, 2))
                attn_weights = torch.softmax(scores, dim=-1)
                context = torch.bmm(attn_weights, all_h)
                combined = torch.cat([gru_output, context], dim=-1)
                combined = torch.tanh(self.combine(combined))
                logits = self.fc(combined[0, -1]) / temperature
                probs = torch.softmax(logits, dim=0)
                next_char_idx = torch.multinomial(probs, 1).item()
                if next_char_idx == EOS_IDX: break
                generated.append(next_char_idx)
                current_char = torch.tensor([[next_char_idx]], device=device)
        return ''.join([idx_to_char[str(idx)] for idx in generated])

# --- Evaluation Logic ---

def run_evaluation():
    print("NAME GENERATION EVALUATION SCRIPT")

    # 1. Load Data
    vocab_path = os.path.join(OUTPUT_DIR, "char_vocab.json")
    names_path = os.path.join(OUTPUT_DIR, "TrainingNames.txt")

    if not os.path.exists(vocab_path) or not os.path.exists(names_path):
        print("Error: Models must be trained using 'name_generation.py' first.")
        return

    with open(vocab_path, 'r') as f:
        char_to_idx = json.load(f)
    idx_to_char = {str(v): k for k, v in char_to_idx.items()}
    vocab_size = len(char_to_idx)

    with open(names_path, 'r') as f:
        training_names = set(line.strip().lower() for line in f if line.strip())

    # 2. Reconstruct Models
    embedding_dim, hidden_size = 64, 128
    
    # Updated configs: (Model Class, Local Filename, Raw Download URL)
    model_configs = {
        'VanillaRNN': (VanillaRNN, "vanilla_rnn.pkl", "https://raw.githubusercontent.com/harshil-1402/NLU-assignment2/main/vanilla_rnn.pkl"),
        'BiLSTM': (BiLSTMGenerator, "bilstm.pkl", "https://raw.githubusercontent.com/harshil-1402/NLU-assignment2/main/bilstm.pkl"),
        'RNNAttention': (RNNAttention, "rnn_attention.pkl", "https://raw.githubusercontent.com/harshil-1402/NLU-assignment2/main/rnn_attention.pkl")
    }

    results = {}
    for name, (model_class, filename, url) in model_configs.items():
        weight_path = os.path.join(OUTPUT_DIR, filename)
        
        # Check if the file exists locally. If not, download it!
        if not os.path.exists(weight_path):
            print(f"Downloading weights for {name}...")
            try:
                urllib.request.urlretrieve(url, weight_path)
            except Exception as e:
                print(f"Skipping {name}: Failed to download weights. Error: {e}")
                continue

        model = model_class(vocab_size, embedding_dim, hidden_size).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        model.eval()

        # 3. Generate and Evaluate
        generated = []
        for _ in range(100):
            generated.append(model.generate(char_to_idx, idx_to_char))

        novel_count = sum(1 for n in generated if n.lower() not in training_names)
        diversity = len(set(generated))

        results[name] = {
            'novelty': novel_count,
            'diversity': diversity,
            'samples': generated[:10]
        }

    # 4. Print Results Table
    print("\nQuantitative Metrics (N=100):")
    print("-" * 50)
    print(f"{'Model':<20} {'Novelty %':<15} {'Diversity %':<15}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<20} {r['novelty']:<15.1f} {r['diversity']:<15.1f}")
    print("-" * 50)

    # 5. Print Qualitative Samples
    print("\nQualitative Samples:")
    for name, r in results.items():
        print(f"\n{name}:")
        print(f"  {', '.join(r['samples'])}")

if __name__ == "__main__":
    run_evaluation()
