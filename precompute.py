import json
import numpy as np
from nltk.corpus import wordnet, stopwords
from nltk.probability import FreqDist
import nltk
import os
import torch
from tqdm import tqdm
import joblib
import inflect
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MODEL_NAME = 'all-mpnet-base-v2'
VECTOR_DIMENSIONS = 768 
OUTPUT_VECTORS_PATH = "data/word_vectors.joblib"
# ... (other output paths are the same)
OUTPUT_VOCAB_PATH = "data/game_vocabulary.json"
OUTPUT_LEMMA_MAP_PATH = "data/lemma_map.joblib"
OUTPUT_COMMON_WORDS_PATH = "data/common_words.json"
OUTPUT_ANTONYM_MAP_PATH = "data/antonym_map.joblib"
OUTPUT_ANNOY_INDEX_PATH = "data/word_vectors.ann"
OUTPUT_ID_TO_WORD_PATH = "data/id_to_word.json"
COMPRESSION_LEVEL = 0

MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 10
NUM_COMMON_WORDS_TO_SKIP = 250

def precompute_all():
    print("="*50); print("  Starting Final SBERT Precomputation Pipeline"); print("="*50)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'; print(f"Using device: {device.upper()}")
    for corpus in ['wordnet', 'stopwords']: nltk.download(corpus, quiet=True)
    
    print(f"\nStep 1: Loading Sentence-BERT model '{MODEL_NAME}'...");
    sbert_model = SentenceTransformer(MODEL_NAME, device=device)
    
    # =====================================================================
    # CHANGE: Use the SBERT model's own vocabulary for a much richer word list
    # =====================================================================
    print("\nStep 2: Filtering the model's full vocabulary...")
    # The tokenizer's vocab is the most reliable source of words the model knows.
    model_vocab = sbert_model.tokenizer.get_vocab().keys()
    english_stopwords = set(stopwords.words('english'))
    
    initial_vocab = {
        word for word in model_vocab 
        if word.isalpha() and word.isascii() and 
        MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH and 
        word not in english_stopwords
    }
    print(f"Found {len(initial_vocab)} suitable words in the model's vocabulary.")

    # We still use brown for frequency analysis as it's a balanced corpus
    print("\nStep 3: Calculating word frequencies...")
    freq_dist = FreqDist(w.lower() for w in nltk.corpus.brown.words())
    sorted_vocab = sorted(list(initial_vocab), key=lambda word: freq_dist[word], reverse=True)
    
    print(f"\nStep 4: Generating {VECTOR_DIMENSIONS}-dimensional vectors with SBERT...")
    embeddings = sbert_model.encode(sorted_vocab, show_progress_bar=True, convert_to_numpy=True)
    word_vectors = {word: embeddings[i].astype(np.float16) for i, word in enumerate(sorted_vocab)}
    if "file" in word_vectors: del word_vectors["file"]
    
    os.makedirs(os.path.dirname(OUTPUT_VECTORS_PATH), exist_ok=True)
    
    print("\nStep 5: Building Annoy index and data maps...")
    word_to_id = {word: i for i, word in enumerate(word_vectors.keys())}
    id_to_word = {i: word for word, i in word_to_id.items()}
    annoy_index = AnnoyIndex(VECTOR_DIMENSIONS, 'angular');
    for i, word in id_to_word.items(): annoy_index.add_item(i, word_vectors[word])
    annoy_index.build(100); annoy_index.save(OUTPUT_ANNOY_INDEX_PATH)
    
    p = inflect.engine(); lemma_map = {}
    for word in word_vectors.keys():
        lemma_map[word] = word; plural = p.plural(word)
        if plural != word: lemma_map[plural] = word

    antonym_map = {}
    for word in tqdm(word_vectors.keys(), desc="Finding Antonyms"):
        antonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                for ant in l.antonyms():
                    if ant.name() in word_vectors: antonyms.add(ant.name())
        if antonyms: antonym_map[word] = list(antonyms)

    print("\nStep 6: Saving final data files...")
    joblib.dump(word_vectors, OUTPUT_VECTORS_PATH, compress=COMPRESSION_LEVEL, protocol=4)
    joblib.dump(lemma_map, OUTPUT_LEMMA_MAP_PATH, compress=COMPRESSION_LEVEL, protocol=4)
    joblib.dump(antonym_map, OUTPUT_ANTONYM_MAP_PATH, compress=COMPRESSION_LEVEL, protocol=4)
    with open(OUTPUT_ID_TO_WORD_PATH, 'w') as f: json.dump(id_to_word, f)
    
    common_words_to_skip = sorted_vocab[:NUM_COMMON_WORDS_TO_SKIP]
    with open(OUTPUT_COMMON_WORDS_PATH, 'w') as f: json.dump(common_words_to_skip, f)
    
    easy_words = [w for w in sorted_vocab[:7000] if w in word_vectors]
    medium_words = [w for w in sorted_vocab[7000:15000] if w in word_vectors]
    hard_words = [w for w in sorted_vocab[15000:] if w in word_vectors]
    final_word_buckets = {"easy": easy_words, "medium": medium_words, "hard": hard_words}
    with open(OUTPUT_VOCAB_PATH, 'w') as f: json.dump(final_word_buckets, f)
    
    print("\n" + "="*50); print("✅ Final Precomputation complete!"); print("="*50)

if __name__ == "__main__":
    precompute_all()