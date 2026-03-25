# NLU Assignment: Word2Vec and Character-Level Name Generation

This project implements two core Natural Language Understanding (NLU) tasks:
1.  **Word2Vec from Scratch**: Training CBOW and Skip-gram models on an IIT Jodhpur corpus.
2.  **Character-Level Name Generation**: Using RNN, BiLSTM, and RNN with Attention to generate Indian names.

## Prerequisites

Ensure you have Python 3.8+ and the following libraries installed:

```bash
pip install torch numpy matplotlib requests beautifulsoup4 scikit-learn gensim wordcloud
```

## Project Structure

*   `word2vec.py`: Main script for Word2Vec implementing CBOW and Skip-gram from scratch.
*   `name_generation.py`: Script for character-level name generation using three different RNN architectures.
*   `train_300d.py`: Helper script to train a 300-dimensional Word2Vec model efficiently.
*   `cleaned_corpus.txt`: Preprocessed text corpus from IIT Jodhpur web pages.
*   `TrainingNames.txt`: Dataset containing 1000 Indian names for character-level generation.

---

## Task 1: Word2Vec (IIT Jodhpur Corpus)

The `word2vec.py` script performs web scraping, data cleaning, and trains multiple embedding models.

### How to Run:
```bash
python3 word2vec.py
```

### Outputs:
*   `wordcloud.png`: Visualization of the most frequent words in the scraped corpus.
*   `vocab.json`: Vocabulary mapping from words to indices.
*   `sg_embeddings.npy` / `cbow_embeddings.npy`: Trained word embedding matrices.
*   `skipgram_tsne.png`: 2D visualization of learned embeddings using t-SNE.

---

## Task 2: Character-Level Name Generation

The `name_generation.py` script trains three models on a dataset of 1000 Indian names.

### How to Run:
```bash
python3 name_generation.py
```

### Models Implemented:
1.  **Vanilla RNN**: A standard recurrent neural network for next-character prediction.
2.  **BiLSTM**: A bidirectional LSTM that captures context from both directions during training.
3.  **RNN with Attention**: Incorporates a causal self-attention mechanism to remember longer dependencies.

### Outputs:
The script will print the architecture details, parameter counts, and generate representative sample names in the terminal.

---

## Summary of Results

*   **Word2Vec**: Successfully captures semantic relationships such as `research` $\approx$ `scholars` and `student` $\approx$ `supervisor`.
*   **Name Generation**: The Attention-based model typically produces the most diverse and natural-sounding names, while the Vanilla RNN handles shorter, simpler patterns.

---

### Note on 300D Embeddings
If you specifically need to train and extract a 300-dimensional vector (as per assignment requirements), use the helper script:
```bash
python3 train_300d.py
```
This will generate `sg_embeddings_300d.npy` and `vocab_300d.json`.
# NLU-PA-2
# NLU-assignment2
