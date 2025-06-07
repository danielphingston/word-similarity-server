import json
import numpy as np
import gensim.downloader as api
from nltk.stem import WordNetLemmatizer
from nltk.corpus import brown
from nltk.probability import FreqDist
import nltk
import os
import torch
from tqdm import tqdm
import joblib # <-- IMPORT JOBLIB

# ================================
# Configuration
# ================================
MODEL_NAME = "glove-wiki-gigaword-300"
# CHANGE: Point to the new file type
OUTPUT_VECTORS_PATH = "data/word_vectors.joblib"
OUTPUT_VOCAB_PATH = "data/game_vocabulary.json"

MIN_WORD_LENGTH = 4
MAX_WORD_LENGTH = 10
EASY_WORDS_COUNT = 2000
MEDIUM_WORDS_COUNT = 5000
HARD_WORDS_COUNT = 10000

def check_gpu():
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ No GPU detected. Running on CPU.")

def precompute_all():
    print("="*50)
    print("  Starting Full Precomputation Pipeline")
    print("="*50)
    
    check_gpu()

    print("\nStep 1: Downloading required NLTK corpora ('brown', 'wordnet')...")
    nltk.download('brown')
    nltk.download('wordnet')

    print(f"\nStep 2: Loading large GloVe model '{MODEL_NAME}'...")
    model = api.load(MODEL_NAME)
    print("Model loaded successfully.")

    print("\nStep 3: Extracting and filtering vocabulary from the GloVe model...")
    initial_vocab = {
        word for word in model.index_to_key
        if MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH and word.isalpha()
    }
    print(f"Found {len(initial_vocab)} suitable words in the model's vocabulary.")

    print("\nStep 4: Calculating word frequencies to determine difficulty...")
    freq_dist = FreqDist(w.lower() for w in brown.words())
    sorted_vocab = sorted(initial_vocab, key=lambda word: freq_dist[word], reverse=True)
    print("Word frequency sorting complete.")

    print("\nStep 5: Assigning words to difficulty buckets...")
    easy_words = set(sorted_vocab[0:EASY_WORDS_COUNT])
    medium_words = set(sorted_vocab[EASY_WORDS_COUNT : EASY_WORDS_COUNT + MEDIUM_WORDS_COUNT])
    hard_words = set(sorted_vocab[EASY_WORDS_COUNT + MEDIUM_WORDS_COUNT :])
    final_game_words = easy_words | medium_words | hard_words
    print(f"Easy: {len(easy_words)}, Medium: {len(medium_words)}, Hard: {len(hard_words)} words.")

    print("\nStep 6: Lemmatizing vocabulary and extracting vectors...")
    lemmatizer = WordNetLemmatizer()
    word_vectors = {}
    lemmatized_game_vocab = set()
    
    for word in tqdm(final_game_words, desc="Lemmatizing & Extracting Vectors"):
        lemma = lemmatizer.lemmatize(word, pos='n')
        if lemma in model and lemma not in word_vectors:
            word_vectors[lemma] = model[lemma]
            lemmatized_game_vocab.add(lemma)
        elif word not in word_vectors: 
             word_vectors[word] = model[word]
             lemmatized_game_vocab.add(word)

    print(f"Extracted {len(word_vectors)} unique vectors.")

    if "file" in word_vectors:
        del word_vectors["file"]
        lemmatized_game_vocab.remove("file")
        print("Note: Removed problematic key 'file' from vocabulary.")

    os.makedirs(os.path.dirname(OUTPUT_VECTORS_PATH), exist_ok=True)
    
    # =====================================================================
    # CHANGE: Use joblib.dump for much faster loading in the API.
    # =====================================================================
    print(f"\nStep 7: Saving {len(word_vectors)} word vectors to '{OUTPUT_VECTORS_PATH}'...")
    joblib.dump(word_vectors, OUTPUT_VECTORS_PATH, compress=3) # compress is optional but good

    print(f"\nStep 8: Saving final game vocabulary to '{OUTPUT_VOCAB_PATH}'...")
    final_word_buckets = {
        "easy": sorted([w for w in easy_words if w in lemmatized_game_vocab]),
        "medium": sorted([w for w in medium_words if w in lemmatized_game_vocab]),
        "hard": sorted([w for w in hard_words if w in lemmatized_game_vocab]),
    }

    with open(OUTPUT_VOCAB_PATH, 'w') as f:
        json.dump(final_word_buckets, f, indent=2)

    print("\n" + "="*50)
    print("✅ Precomputation complete!")
    print("="*50)

if __name__ == "__main__":
    precompute_all()