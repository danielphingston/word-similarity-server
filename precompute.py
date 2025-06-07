import json
import numpy as np
import gensim.downloader as api
from nltk.stem import WordNetLemmatizer
from nltk.corpus import brown, stopwords, wordnet
from nltk.probability import FreqDist
import nltk
import os
import torch
from tqdm import tqdm
import joblib
import inflect
from annoy import AnnoyIndex # <-- IMPORT Annoy

# --- Configuration ---
MODEL_NAME = "glove-wiki-gigaword-300"
OUTPUT_VECTORS_PATH = "data/word_vectors.joblib"
OUTPUT_VOCAB_PATH = "data/game_vocabulary.json"
OUTPUT_LEMMA_MAP_PATH = "data/lemma_map.joblib"
OUTPUT_COMMON_WORDS_PATH = "data/common_words.json"
OUTPUT_ANTONYM_MAP_PATH = "data/antonym_map.joblib"
# NEW: Path for our Annoy index for fast hint lookups
OUTPUT_ANNOY_INDEX_PATH = "data/word_vectors.ann"
# NEW: Path for mapping Annoy IDs back to words
OUTPUT_ID_TO_WORD_PATH = "data/id_to_word.json"

VECTOR_DIMENSIONS = 300 # Must match the model
MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 10
NUM_COMMON_WORDS_TO_SKIP = 250

def precompute_all():
    # ... (Steps 1-5 are the same as before) ...
    print("="*50); print("  Starting Full Precomputation Pipeline"); print("="*50)
    print("\nStep 1: NLTK Corpora..."); nltk.download('brown', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('stopwords', quiet=True)
    print(f"\nStep 2: Loading GloVe model '{MODEL_NAME}'..."); model = api.load(MODEL_NAME)
    print("\nStep 3: Filtering vocabulary..."); english_stopwords = set(stopwords.words('english'))
    initial_vocab = {w for w in model.index_to_key if w.isalpha() and w.isascii() and MIN_WORD_LENGTH <= len(w) <= MAX_WORD_LENGTH and w not in english_stopwords}
    print(f"Found {len(initial_vocab)} suitable words.")
    print("\nStep 4: Calculating frequencies..."); freq_dist = FreqDist(w.lower() for w in brown.words()); sorted_vocab = sorted(initial_vocab, key=lambda word: freq_dist[word], reverse=True)
    final_game_words = set(sorted_vocab); print(f"Total game words considered: {len(final_game_words)}")
    print("\nStep 5: Extracting vectors..."); p = inflect.engine()
    word_vectors = {}; lemma_map = {}
    for word in tqdm(final_game_words, desc="Extracting & Mapping"):
        if word in model:
            word_vectors[word] = model[word].astype(np.float16)
            lemma_map[word] = word
            plural = p.plural(word)
            if plural and plural != word: lemma_map[plural] = word
    print(f"Processed {len(word_vectors)} unique words.")
    if "file" in word_vectors: del word_vectors["file"]
    os.makedirs(os.path.dirname(OUTPUT_VECTORS_PATH), exist_ok=True)
    
    # =====================================================================
    # NEW Step 6: Build and save the Annoy index for fast searching
    # =====================================================================
    print("\nStep 6: Building Annoy index for fast hint searches...")
    # Create mappings between integer IDs and words, which Annoy needs
    word_to_id = {word: i for i, word in enumerate(word_vectors.keys())}
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    # Create and build the index
    annoy_index = AnnoyIndex(VECTOR_DIMENSIONS, 'angular') # 'angular' is cosine similarity
    for word, i in tqdm(word_to_id.items(), desc="Adding to Annoy Index"):
        annoy_index.add_item(i, word_vectors[word])
        
    annoy_index.build(100) # 100 trees for a good balance of speed and accuracy
    
    # Save the index and the id->word map
    annoy_index.save(OUTPUT_ANNOY_INDEX_PATH)
    with open(OUTPUT_ID_TO_WORD_PATH, 'w') as f:
        json.dump(id_to_word, f)
    print(f"Saved Annoy index and ID map.")
    
    # ... (The rest of the script is the same, just re-numbered) ...
    print("\nStep 7: Creating antonym map..."); antonym_map = {}
    for word in tqdm(word_vectors.keys(), desc="Finding Antonyms"):
        antonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                for ant in l.antonyms():
                    if ant.name() in word_vectors: antonyms.add(ant.name())
        if antonyms: antonym_map[word] = list(antonyms)
    joblib.dump(antonym_map, OUTPUT_ANTONYM_MAP_PATH, compress=3, protocol=4)

    print("\nStep 8: Saving final data files...")
    joblib.dump(word_vectors, OUTPUT_VECTORS_PATH, compress=3, protocol=4)
    joblib.dump(lemma_map, OUTPUT_LEMMA_MAP_PATH, compress=3, protocol=4)
    common_words_to_skip = sorted_vocab[:NUM_COMMON_WORDS_TO_SKIP]
    with open(OUTPUT_COMMON_WORDS_PATH, 'w') as f: json.dump(common_words_to_skip, f)
    final_word_buckets = {"easy": [w for w in sorted_vocab[:5000] if w in word_vectors], "medium": [w for w in sorted_vocab[5000:12000] if w in word_vectors], "hard": [w for w in sorted_vocab[12000:] if w in word_vectors]}
    with open(OUTPUT_VOCAB_PATH, 'w') as f: json.dump(final_word_buckets, f)
        
    print("✅ Precomputation complete!")

if __name__ == "__main__":
    precompute_all()