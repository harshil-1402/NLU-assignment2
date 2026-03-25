import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from itertools import chain

# For web scraping
import requests
from bs4 import BeautifulSoup

# For visualization
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# For dimensionality reduction (Task 4)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# For comparison with existing Word2Vec implementation
from gensim.models import Word2Vec as GensimWord2Vec

# Reproducibility: set seeds so that training results are consistent across runs
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Output directory for saving plots and corpus
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TASK 1: DATASET PREPARATION

print("TASK 1: DATASET PREPARATION")

#  1a. Data Collection 
# We scrape text from several IIT Jodhpur web pages covering departments,
# academics, research, and campus information.

URLS = [
    "https://iitj.ac.in/",
    "https://iitj.ac.in/es/en/engineering-science",
    "https://iitj.ac.in/department/index.php?id=dept_structure",
    "https://iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    "https://cse.iitj.ac.in/",
    "https://ee.iitj.ac.in/",
    "https://iitj.ac.in/office-of-students/en/office-of-students",
    "https://iitj.ac.in/office-of-executive-education/en/office-of-executive-education",
    "https://iitj.ac.in/main/en/visiting-faculty-members",
    "https://iitj.ac.in/dia/en/dia",
    "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    "https://iitj.ac.in/PageImages/Gallery/03-2025/1_Academic_Regulations_Final_03_09_2019.pdf",
    "https://iitj.ac.in/PageImages/Gallery/03-2025/2_Academic_Regulations_Final_03_09_2019.pdf",
    "https://iitj.ac.in/PageImages/Gallery/03-2025/2.1_notification_26102020.pdf",
    "https://iitj.ac.in/PageImages/Gallery/03-2025/3_Academic_Regulations_Final_03_09_2019.pdf",
]


def scrape_text_from_url(url, timeout=10):
    
    try:
        # Send HTTP request with a reasonable timeout to avoid hanging
        response = requests.get(url, timeout=timeout, verify=False)
        response.raise_for_status()

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements that contain non-content text
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Extract all visible text from the parsed HTML
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        print(f"  [WARNING] Could not scrape {url}: {e}")
        return ""


# Bypass scraping and load from existing corpus
corpus_path = os.path.join(OUTPUT_DIR, "cleaned_corpus.txt")
raw_documents = [] # Placeholder to avoid NameError
if os.path.exists(corpus_path):
    print(f"\n[1] Loading existing corpus from: {corpus_path}")
    with open(corpus_path, 'r') as f:
        corpus_tokens = f.read().split()
else:
    print(f"\n[ERROR] {corpus_path} not found. Cannot proceed without corpus.")
    exit(1)

# Build vocabulary: map each unique word to an integer index
# This is essential for creating the embedding lookup table
word_counts = Counter(corpus_tokens)

# Filter out very rare words (appearing only once) to reduce noise
# Rare words don't have enough context to learn good embeddings
MIN_FREQ = 2
vocab = {word: idx for idx, (word, count) in enumerate(
    sorted(word_counts.items(), key=lambda x: -x[1])
) if count >= MIN_FREQ}

# Add an <UNK> token for words not in vocabulary
vocab['<UNK>'] = len(vocab)
idx_to_word = {idx: word for word, idx in vocab.items()}

# Replace rare words in corpus with <UNK>
corpus_indices = [vocab.get(w, vocab['<UNK>']) for w in corpus_tokens]

# -- 1c. Report Dataset Statistics --
print("\n[1c] Dataset Statistics:")
# print(f"  Total number of documents:  {len(raw_documents)}")
print(f"  Total tokens (after preprocessing): {len(corpus_tokens)}")
print(f"  Vocabulary size (min_freq={MIN_FREQ}): {len(vocab)}")
print(f"  Most frequent words:")
for word, count in word_counts.most_common(15):
    print(f"    {word:20s} -> {count}")

# Save the cleaned corpus to a text file for reference
corpus_path = os.path.join(OUTPUT_DIR, "cleaned_corpus.txt")
with open(corpus_path, 'w') as f:
    f.write(' '.join(corpus_tokens))
print(f"\n  Cleaned corpus saved to: {corpus_path}")

# 1d. Word Cloud 
# Generate a word cloud from the cleaned corpus to visualize most frequent words
print("\n[1d] Generating Word Cloud...")
wordcloud = WordCloud(
    width=800, height=400,
    background_color='white',
    max_words=100,
    colormap='viridis',
    contour_width=1,
    contour_color='steelblue'
).generate(' '.join(corpus_tokens))

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud — IIT Jodhpur Corpus', fontsize=16)
plt.tight_layout()
wordcloud_path = os.path.join(OUTPUT_DIR, "wordcloud.png")
plt.savefig(wordcloud_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Word cloud saved to: {wordcloud_path}")


# TASK 2: MODEL TRAINING

print("TASK 2: MODEL TRAINING")

# Determine the device to use: GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Using device: {device}")

VOCAB_SIZE = len(vocab)


# 2a. CBOW Model (Continuous Bag of Words) — From Scratch
# CBOW predicts the center word given its surrounding context words.
# Architecture: Embedding layer → Average pooling → Linear projection
# Loss: CrossEntropyLoss (multi-class classification over vocabulary)

class CBOWModel(nn.Module):
    

    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        # Embedding layer maps each word index to a dense vector
        # These are the word embeddings we want to learn
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Linear layer projects the averaged context embedding to vocabulary
        # size for predicting the center word
        self.linear = nn.Linear(embedding_dim, vocab_size)

        # Initialize weights with small random values for stable training
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, context_indices):
        
        # Look up embeddings for all context words
        # Shape: (batch_size, 2 * window_size, embedding_dim)
        embeds = self.embeddings(context_indices)

        # Average the context word embeddings to get a single representation
        # This is the key idea of CBOW: the context is summarized by averaging
        # Shape: (batch_size, embedding_dim)
        mean_embed = embeds.mean(dim=1)

        # Project to vocabulary size for classification
        # Shape: (batch_size, vocab_size)
        out = self.linear(mean_embed)
        return out


def create_cbow_dataset(corpus_indices, window_size):
    
    data = []
    for i in range(window_size, len(corpus_indices) - window_size):
        # Gather context words: window_size words before and after the target
        context = []
        for j in range(i - window_size, i + window_size + 1):
            if j != i:  # Exclude the target word itself from context
                context.append(corpus_indices[j])
        target = corpus_indices[i]
        data.append((context, target))
    return data


# 2b. Skip-gram with Negative Sampling — From Scratch

# Skip-gram predicts context words given the center word.
# With negative sampling, we train a binary classifier to distinguish
# real (center, context) pairs from randomly sampled negative pairs.
# Loss: Binary Cross-Entropy with Logits

class SkipGramNS(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNS, self).__init__()
        # Two separate embedding layers: one for center words, one for context
        # This separation allows the model to learn different representations
        # for when a word appears as center vs. context
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with small random values
        nn.init.xavier_uniform_(self.center_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)

    def forward(self, center_idx, context_idx):
        
        # Look up center word embeddings: (batch_size, embedding_dim)
        center_embed = self.center_embeddings(center_idx)
        # Look up context word embeddings: (batch_size, embedding_dim)
        context_embed = self.context_embeddings(context_idx)

        # Dot product measures similarity between center and context vectors
        # Shape: (batch_size,)
        score = (center_embed * context_embed).sum(dim=1)
        return score


def create_skipgram_dataset(corpus_indices, window_size):
    
    data = []
    for i in range(window_size, len(corpus_indices) - window_size):
        center = corpus_indices[i]
        # Pair the center word with each context word in the window
        for j in range(i - window_size, i + window_size + 1):
            if j != i:  # Skip the center word itself
                data.append((center, corpus_indices[j]))
    return data


def get_negative_sampling_distribution(corpus_indices, vocab_size):
    
    # Count frequency of each word in the corpus
    freq = np.zeros(vocab_size)
    for idx in corpus_indices:
        freq[idx] += 1

    # Raise to power 0.75 as suggested by Mikolov et al. (2013)
    # This smooths the distribution, making rare words more likely
    # to be selected as negatives
    freq = freq ** 0.75

    # Normalize to create a valid probability distribution
    freq = freq / freq.sum()
    return freq


# 2c. Training Loop

def train_cbow(corpus_indices, vocab_size, embedding_dim, window_size,
               num_epochs=30, lr=0.01, batch_size=256):
    
    print(f"    Training CBOW (dim={embedding_dim}, win={window_size})...")

    # Create training pairs: (context_words, target_word)
    dataset = create_cbow_dataset(corpus_indices, window_size)
    if len(dataset) == 0:
        print("    [WARNING] No training data created. Corpus too small.")
        model = CBOWModel(vocab_size, embedding_dim).to(device)
        return model, float('inf')

    # Initialize model, loss function, and optimizer
    model = CBOWModel(vocab_size, embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop with mini-batches
    model.train()
    final_loss = 0.0
    for epoch in range(num_epochs):
        # Shuffle data for each epoch to avoid learning order-dependent patterns
        random.shuffle(dataset)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start + batch_size]
            # Separate contexts and targets, convert to tensors
            contexts = torch.tensor([pair[0] for pair in batch],
                                    dtype=torch.long, device=device)
            targets = torch.tensor([pair[1] for pair in batch],
                                   dtype=torch.long, device=device)

            # Forward pass: predict target from context
            output = model(contexts)
            loss = criterion(output, targets)

            # Backward pass: compute gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        final_loss = total_loss / max(n_batches, 1)
        # Print progress every 10 epochs to track convergence
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch + 1}/{num_epochs}, Loss: {final_loss:.4f}")

    return model, final_loss


def train_skipgram(corpus_indices, vocab_size, embedding_dim, window_size,
                   num_negatives=5, num_epochs=30, lr=0.01, batch_size=256):
    
    print(f"    Training Skip-gram (dim={embedding_dim}, win={window_size}, "
          f"neg={num_negatives})...")

    # Create center-context pairs
    dataset = create_skipgram_dataset(corpus_indices, window_size)
    if len(dataset) == 0:
        print("    [WARNING] No training data. Corpus too small.")
        model = SkipGramNS(vocab_size, embedding_dim).to(device)
        return model, float('inf')

    # Compute the negative sampling distribution (unigram^0.75)
    neg_dist = get_negative_sampling_distribution(corpus_indices, vocab_size)

    # Initialize model and optimizer
    model = SkipGramNS(vocab_size, embedding_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    final_loss = 0.0
    for epoch in range(num_epochs):
        random.shuffle(dataset)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(dataset), batch_size):
            batch = dataset[start:start + batch_size]
            centers = [pair[0] for pair in batch]
            contexts = [pair[1] for pair in batch]
            curr_batch_size = len(batch)

            # For each positive pair, sample num_negatives random words
            # These negatives provide the contrastive signal: the model
            # must learn to score real context words higher than random ones
            neg_samples = np.random.choice(
                vocab_size, size=(curr_batch_size, num_negatives), p=neg_dist
            )

            # Build training batch: positive pairs (label=1) + negative pairs (label=0)
            all_centers = []
            all_contexts = []
            all_labels = []

            for i in range(curr_batch_size):
                # Positive example: real (center, context) pair
                all_centers.append(centers[i])
                all_contexts.append(contexts[i])
                all_labels.append(1.0)

                # Negative examples: (center, random_word) pairs
                for neg in neg_samples[i]:
                    all_centers.append(centers[i])
                    all_contexts.append(int(neg))
                    all_labels.append(0.0)

            # Convert to tensors and move to device
            center_t = torch.tensor(all_centers, dtype=torch.long, device=device)
            context_t = torch.tensor(all_contexts, dtype=torch.long, device=device)
            labels_t = torch.tensor(all_labels, dtype=torch.float, device=device)

            # Forward pass: compute similarity scores
            scores = model(center_t, context_t)
            loss = criterion(scores, labels_t)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        final_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch + 1}/{num_epochs}, Loss: {final_loss:.4f}")

    return model, final_loss


# 2d. Extract embeddings from trained models

def get_embeddings_from_cbow(model):
    """Extracts the embedding matrix from a trained CBOW model."""
    return model.embeddings.weight.detach().cpu().numpy()


def get_embeddings_from_skipgram(model):
    
    return model.center_embeddings.weight.detach().cpu().numpy()


# 2e. Hyperparameter experiments
# We try different combinations of embedding dimension, window size,
# and number of negative samples to find the best configuration

EMBEDDING_DIMS = [300]
WINDOW_SIZES = [5]
NEGATIVE_SAMPLES = [10]

print("\n[2] Training Word2Vec models with different hyperparameters...")
print("    (This may take a few minutes)\n")

results = []

# Store best models for later semantic analysis and visualization
best_cbow_model = None
best_cbow_loss = float('inf')
best_cbow_config = {}
best_sg_model = None
best_sg_loss = float('inf')
best_sg_config = {}

for dim in EMBEDDING_DIMS:
    for win in WINDOW_SIZES:
        # Train CBOW with current hyperparameters
        cbow_model, cbow_loss = train_cbow(
            corpus_indices, VOCAB_SIZE, dim, win,
            num_epochs=30, lr=0.01
        )
        # Track the best CBOW model (lowest loss)
        if cbow_loss < best_cbow_loss:
            best_cbow_loss = cbow_loss
            best_cbow_model = cbow_model
            best_cbow_config = {'dim': dim, 'win': win}

        for neg in NEGATIVE_SAMPLES:
            # Train Skip-gram with current hyperparameters
            sg_model, sg_loss = train_skipgram(
                corpus_indices, VOCAB_SIZE, dim, win,
                num_negatives=neg, num_epochs=30, lr=0.01
            )
            # Track the best Skip-gram model
            if sg_loss < best_sg_loss:
                best_sg_loss = sg_loss
                best_sg_model = sg_model
                best_sg_config = {'dim': dim, 'win': win, 'neg': neg}

            results.append({
                'Model': 'Skip-gram NS',
                'Embed Dim': dim,
                'Window': win,
                'Neg Samples': neg,
                'Final Loss': f"{sg_loss:.4f}"
            })

        results.append({
            'Model': 'CBOW',
            'Embed Dim': dim,
            'Window': win,
            'Neg Samples': '-',
            'Final Loss': f"{cbow_loss:.4f}"
        })

# Print results table
print("\n  Hyperparameter Experiment Results:")
print("  " + "-" * 65)
print(f"  {'Model':<15} {'Dim':<6} {'Win':<6} {'Neg':<6} {'Loss':<10}")
print("  " + "-" * 65)
for r in results:
    print(f"  {r['Model']:<15} {r['Embed Dim']:<6} {r['Window']:<6} "
          f"{r['Neg Samples']:<6} {r['Final Loss']:<10}")
print("  " + "-" * 65)

print(f"\n  Best CBOW config:     dim={best_cbow_config['dim']}, "
      f"win={best_cbow_config['win']}, loss={best_cbow_loss:.4f}")
print(f"  Best Skip-gram config: dim={best_sg_config['dim']}, "
      f"win={best_sg_config['win']}, neg={best_sg_config['neg']}, "
      f"loss={best_sg_loss:.4f}")


# 2f. Train Gensim models for comparison
# We train gensim Word2Vec with the same hyperparameters as our best scratch
# models to enable a fair comparison

print("\n[2f] Training Gensim Word2Vec models for comparison...")

# Gensim expects a list of sentences (lists of words)
gensim_sentences = processed_docs

# CBOW comparison (sg=0 means CBOW in gensim)
gensim_cbow = GensimWord2Vec(
    sentences=gensim_sentences,
    vector_size=best_cbow_config['dim'],
    window=best_cbow_config['win'],
    min_count=MIN_FREQ,
    sg=0,  # CBOW mode
    epochs=30,
    seed=SEED,
    workers=1  # Single thread for reproducibility
)
print(f"  ✓ Gensim CBOW trained (dim={best_cbow_config['dim']}, "
      f"win={best_cbow_config['win']})")

# Skip-gram comparison (sg=1 means Skip-gram in gensim)
gensim_sg = GensimWord2Vec(
    sentences=gensim_sentences,
    vector_size=best_sg_config['dim'],
    window=best_sg_config['win'],
    min_count=MIN_FREQ,
    sg=1,  # Skip-gram mode
    negative=best_sg_config['neg'],
    epochs=30,
    seed=SEED,
    workers=1
)
print(f"  ✓ Gensim Skip-gram trained (dim={best_sg_config['dim']}, "
      f"win={best_sg_config['win']}, neg={best_sg_config['neg']})")


# TASK 3: SEMANTIC ANALYSIS

print("TASK 3: SEMANTIC ANALYSIS")


# Extract embedding matrices from our best models
cbow_embeddings = get_embeddings_from_cbow(best_cbow_model)
sg_embeddings = get_embeddings_from_skipgram(best_sg_model)

# Save best embeddings and vocabulary for Task P1
print("\n[3] Saving models and vocabulary...")
np.save(os.path.join(OUTPUT_DIR, "cbow_embeddings.npy"), cbow_embeddings)
np.save(os.path.join(OUTPUT_DIR, "sg_embeddings.npy"), sg_embeddings)
import json
with open(os.path.join(OUTPUT_DIR, "vocab.json"), "w") as f:
    json.dump(vocab, f)
print(f"    Saved: cbow_embeddings.npy, sg_embeddings.npy, vocab.json")


def cosine_similarity(vec_a, vec_b):
    
    # Handle zero vectors to avoid division by zero
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)


def find_nearest_neighbors(word, embeddings, vocab, idx_to_word, top_k=5):
    
    if word not in vocab:
        return [(f"'{word}' not in vocabulary", 0.0)]

    word_idx = vocab[word]
    word_vec = embeddings[word_idx]

    # Compute cosine similarity between the query word and all other words
    similarities = []
    for idx in range(len(embeddings)):
        if idx == word_idx:
            continue  # Skip the word itself
        sim = cosine_similarity(word_vec, embeddings[idx])
        similarities.append((idx, sim))

    # Sort by similarity in descending order and take top-k
    similarities.sort(key=lambda x: -x[1])
    return [(idx_to_word[idx], sim) for idx, sim in similarities[:top_k]]


# -- 3a. Top-5 Nearest Neighbors --
# Report nearest neighbors for the specified query words
query_words = ['research', 'student', 'phd', 'examination']

print("\n[3a] Top-5 Nearest Neighbors:")

for word in query_words:
    print(f"\n  Query: '{word}'")

    # From-scratch CBOW
    neighbors = find_nearest_neighbors(word, cbow_embeddings, vocab, idx_to_word)
    print(f"    CBOW (scratch):    ", end="")
    print(", ".join([f"{w} ({s:.3f})" for w, s in neighbors]))

    # From-scratch Skip-gram
    neighbors = find_nearest_neighbors(word, sg_embeddings, vocab, idx_to_word)
    print(f"    Skip-gram (scratch):", end="")
    print(", ".join([f"{w} ({s:.3f})" for w, s in neighbors]))

    # Gensim CBOW
    if word in gensim_cbow.wv:
        gensim_neighbors = gensim_cbow.wv.most_similar(word, topn=5)
        print(f"    CBOW (gensim):     ", end="")
        print(", ".join([f"{w} ({s:.3f})" for w, s in gensim_neighbors]))
    else:
        print(f"    CBOW (gensim):     '{word}' not in gensim vocabulary")

    # Gensim Skip-gram
    if word in gensim_sg.wv:
        gensim_neighbors = gensim_sg.wv.most_similar(word, topn=5)
        print(f"    Skip-gram (gensim):", end="")
        print(", ".join([f"{w} ({s:.3f})" for w, s in gensim_neighbors]))
    else:
        print(f"    Skip-gram (gensim): '{word}' not in gensim vocabulary")


# -- 3b. Analogy Experiments --
# Analogy: A is to B as C is to ? → vec(B) - vec(A) + vec(C) ≈ vec(?)

def analogy(word_a, word_b, word_c, embeddings, vocab, idx_to_word, top_k=3):
    
    # Check all words are in vocabulary
    for w in [word_a, word_b, word_c]:
        if w not in vocab:
            return [(f"'{w}' not in vocabulary", 0.0)]

    # Compute the analogy vector using vector arithmetic
    vec_a = embeddings[vocab[word_a]]
    vec_b = embeddings[vocab[word_b]]
    vec_c = embeddings[vocab[word_c]]
    result_vec = vec_b - vec_a + vec_c

    # Exclude the input words from results to avoid trivial answers
    exclude = {vocab[word_a], vocab[word_b], vocab[word_c]}

    # Find nearest words to the result vector
    similarities = []
    for idx in range(len(embeddings)):
        if idx in exclude:
            continue
        sim = cosine_similarity(result_vec, embeddings[idx])
        similarities.append((idx, sim))

    similarities.sort(key=lambda x: -x[1])
    return [(idx_to_word[idx], sim) for idx, sim in similarities[:top_k]]


# Define analogy experiments relevant to IIT Jodhpur context
# Format: (A, B, C) where A:B :: C:?
analogies = [
    ('ug', 'btech', 'pg'),         # UG:BTech :: PG:? (expected: MTech)
    ('student', 'exam', 'faculty'),  # student:exam :: faculty:? (expected: research)
    ('department', 'engineering', 'centre'),  # department:engineering :: centre:? (expected: research/technology)
]

print("\n\n[3b] Analogy Experiments:")
print("  (A : B :: C : ?)\n")

for word_a, word_b, word_c in analogies:
    print(f"  {word_a} : {word_b} :: {word_c} : ?")

    # From-scratch CBOW
    answers = analogy(word_a, word_b, word_c, cbow_embeddings, vocab, idx_to_word)
    print(f"    CBOW (scratch):     ", end="")
    print(", ".join([f"{w} ({s:.3f})" for w, s in answers]))

    # From-scratch Skip-gram
    answers = analogy(word_a, word_b, word_c, sg_embeddings, vocab, idx_to_word)
    print(f"    Skip-gram (scratch):", end="")
    print(", ".join([f"{w} ({s:.3f})" for w, s in answers]))

    # Gensim models
    for model_name, gensim_model in [("CBOW (gensim)", gensim_cbow),
                                      ("SG (gensim)  ", gensim_sg)]:
        try:
            if all(w in gensim_model.wv for w in [word_a, word_b, word_c]):
                result = gensim_model.wv.most_similar(
                    positive=[word_b, word_c], negative=[word_a], topn=3
                )
                print(f"    {model_name}:     ", end="")
                print(", ".join([f"{w} ({s:.3f})" for w, s in result]))
            else:
                missing = [w for w in [word_a, word_b, word_c]
                           if w not in gensim_model.wv]
                print(f"    {model_name}:     Words not in vocab: {missing}")
        except Exception as e:
            print(f"    {model_name}:     Error: {e}")
    print()

# Discussion of results
print("\n  --- Discussion ---")
print("  The analogy results reflect the limited corpus size. With a small,")
print("  domain-specific corpus from IIT Jodhpur, analogies may not produce")
print("  the exact expected answers. However, the returned words are typically")
print("  from related semantic fields, demonstrating that the embeddings do")
print("  capture meaningful relationships. The scratch models and gensim models")
print("  may differ in quality due to implementation details and optimization")
print("  strategies in gensim (e.g., better subsampling, learning rate decay).")


# TASK 4: VISUALIZATION

print("TASK 4: VISUALIZATION")

# Define semantic word groups for visualization
# These groups represent different aspects of IIT Jodhpur
word_groups = {
    'Academic Programs': ['btech', 'mtech', 'phd', 'msc', 'programme',
                          'undergraduate', 'postgraduate', 'degree'],
    'Research': ['research', 'publications', 'conference', 'innovation',
                 'projects', 'faculty', 'laboratory'],
    'Departments': ['engineering', 'science', 'department', 'computer',
                    'electrical', 'mechanical'],
    'Campus Life': ['campus', 'hostel', 'library', 'placement',
                    'student', 'clubs', 'sports'],
    'Courses': ['algorithms', 'learning', 'processing',
                'systems', 'design', 'networks'],
}


def visualize_embeddings(embeddings, vocab, word_groups, method, model_name,
                         filename):
    
    # Collect all words that exist in our vocabulary from the groups
    words = []
    group_labels = []
    vectors = []
    for group_name, word_list in word_groups.items():
        for word in word_list:
            if word in vocab:
                words.append(word)
                group_labels.append(group_name)
                vectors.append(embeddings[vocab[word]])

    if len(vectors) < 3:
        print(f"    [WARNING] Not enough words in vocabulary for {model_name}. "
              f"Skipping visualization.")
        return

    vectors = np.array(vectors)

    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=SEED)
        coords = reducer.fit_transform(vectors)
        explained_var = sum(reducer.explained_variance_ratio_) * 100
        title_suffix = f" (PCA, {explained_var:.1f}% var explained)"
    else:  # t-SNE
        # Perplexity must be less than the number of samples
        perp = min(5, len(vectors) - 1)
        reducer = TSNE(n_components=2, random_state=SEED, perplexity=perp,
                       max_iter=1000)
        coords = reducer.fit_transform(vectors)
        title_suffix = " (t-SNE)"

    # Create the scatter plot with color-coded groups
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(word_groups)))
    group_names = list(word_groups.keys())

    for i, (word, coord) in enumerate(zip(words, coords)):
        group_idx = group_names.index(group_labels[i])
        ax.scatter(coord[0], coord[1], color=colors[group_idx], s=80,
                   edgecolors='black', linewidth=0.5, zorder=2)
        # Annotate each point with the word label, slightly offset
        ax.annotate(word, (coord[0], coord[1]),
                    fontsize=8, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    # Add legend for semantic groups
    for i, group_name in enumerate(group_names):
        ax.scatter([], [], color=colors[i], label=group_name, s=80,
                   edgecolors='black', linewidth=0.5)
    ax.legend(loc='best', fontsize=9)

    ax.set_title(f'{model_name}{title_suffix}', fontsize=14)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {filepath}")


print("\n[4] Generating visualizations...")

# Generate PCA and t-SNE plots for both CBOW and Skip-gram models
visualize_embeddings(cbow_embeddings, vocab, word_groups, 'pca',
                     'CBOW (From Scratch)', 'cbow_pca.png')
visualize_embeddings(cbow_embeddings, vocab, word_groups, 'tsne',
                     'CBOW (From Scratch)', 'cbow_tsne.png')
visualize_embeddings(sg_embeddings, vocab, word_groups, 'pca',
                     'Skip-gram (From Scratch)', 'skipgram_pca.png')
visualize_embeddings(sg_embeddings, vocab, word_groups, 'tsne',
                     'Skip-gram (From Scratch)', 'skipgram_tsne.png')

print("\n  --- Interpretation ---")
print("  CBOW tends to produce tighter clusters for frequent words because it")
print("  averages context, which smooths out noise. Skip-gram with negative")
print("  sampling often spreads words more in the embedding space because it")
print("  processes each center-context pair independently, giving each word")
print("  more individual representation. Both models should show some semantic")
print("  grouping — words related to academics, research, departments, and")
print("  campus life should cluster together in the 2D projection.")

print("\n" + "=" * 70)
print("ALL TASKS COMPLETED SUCCESSFULLY")
print("=" * 70)
print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("  - cleaned_corpus.txt")
print("  - wordcloud.png")
print("  - cbow_pca.png, cbow_tsne.png")
print("  - skipgram_pca.png, skipgram_tsne.png")
