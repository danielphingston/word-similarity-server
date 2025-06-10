import json
import logging
import os
import time

import joblib
import nltk
import numpy as np
import torch
from annoy import AnnoyIndex
from nltk.corpus import names, stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from wordfreq import top_n_list

# ================================
# 1. Setup Logging
# ================================
# Configure logging to provide timestamped, informative output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ================================
# 2. Configuration
# ================================
MODEL_NAME = './outputs/outputs/my-custom-mpnet-model-checkpoints/checkpoint-60000'
VECTOR_DIMENSIONS = 768
SBERT_BATCH_SIZE = 10000  # Optimal batch size for faster encoding on GPU

# --- Output Paths ---
OUTPUT_DIR = "data"
OUTPUT_VECTORS_PATH = os.path.join(OUTPUT_DIR, "word_vectors.joblib")
OUTPUT_VOCAB_PATH = os.path.join(OUTPUT_DIR, "game_vocabulary.json")
OUTPUT_COMMON_WORDS_PATH = os.path.join(OUTPUT_DIR, "common_words.json")
OUTPUT_ANNOY_INDEX_PATH = os.path.join(OUTPUT_DIR, "word_vectors.ann")
OUTPUT_ID_TO_WORD_PATH = os.path.join(OUTPUT_DIR, "id_to_word.json")
OUTPUT_LEMMATIZATION_MAP_PATH = os.path.join(OUTPUT_DIR, "lemmatization_map.json")


# --- Data Curation Constants ---
MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 100
NUM_COMMON_WORDS_TO_SKIP = 100
COCA_WORDS_FOR_VOCAB_SIZE = 40000
COCA_WORDS_FOR_RANKING = 40000
COMPRESSION_LEVEL = 0  # A little compression for the joblib file is fine

# --- Dynamic Difficulty Bucketing ---
EASY_PERCENTILE = 0.1   # Top 30% most frequent words are 'easy'
MEDIUM_PERCENTILE = 0.5 # The next 40% are 'medium' (from 30% to 70%)

def load_resources(device):
    """Downloads NLTK data and loads the SentenceTransformer model."""
    logging.info("Step 1: Loading resources...")
    for corpus in ['stopwords', 'names', 'wordnet', 'omw-1.4']:
        nltk.download(corpus, quiet=True)
    logging.info("NLTK corpora are ready.")

    logging.info(f"Loading Sentence-BERT model '{MODEL_NAME}' onto device: {device.upper()}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model

def create_lemmatization_map(source_vocab, lemmatizer, filters):
    """
    Creates a map from inflected words to their lemmas.
    Example: {"walking": "walk", "cities": "city"}
    """
    logging.info("Creating a lemmatization map for API use...")
    lemmatization_map = {}
    english_stopwords, all_names, min_len, max_len = filters

    for word in source_vocab:
        # Check original word against filters first
        if not (word.isalpha() and word.isascii() and
                min_len <= len(word) <= max_len and
                word not in english_stopwords and word not in all_names):
            continue

        # Lemmatize the word
        lemma = lemmatizer.lemmatize(word.lower(), pos=wordnet.VERB)
        
        # If the word is different from its lemma, it's an inflection we should map
        if word.lower() != lemma:
            # Final check to ensure the lemma is a valid game word
            if (min_len <= len(lemma) <= max_len and lemma not in english_stopwords):
                 lemmatization_map[word.lower()] = lemma
    
    logging.info(f"Created a map with {len(lemmatization_map)} inflection-to-lemma pairs.")
    return lemmatization_map

def build_vocabulary(sbert_model):
    """Combines model and COCA vocab, filters, and lemmatizes to create a clean word list."""
    logging.info("Step 2: Building a rich, lemmatized base vocabulary...")

    # Combine sources
    model_vocab = set(sbert_model.tokenizer.get_vocab().keys())
    coca_for_vocab = set(top_n_list('en', COCA_WORDS_FOR_VOCAB_SIZE))
    combined_source_vocab = model_vocab.union(coca_for_vocab)
    logging.info(f"Combined {len(model_vocab)} model words and {len(coca_for_vocab)} COCA words.")

    # Setup filters and lemmatizer
    english_stopwords = set(stopwords.words('english'))
    all_names = set(names.words())
    lemmatizer = WordNetLemmatizer()

    # --- NEW: Create and save the lemmatization map ---
    filters_for_map = (english_stopwords, all_names, MIN_WORD_LENGTH, MAX_WORD_LENGTH)
    lemmatization_map = create_lemmatization_map(combined_source_vocab, lemmatizer, filters_for_map)
    with open(OUTPUT_LEMMATIZATION_MAP_PATH, 'w') as f:
        json.dump(lemmatization_map, f, indent=2)
    # --- END NEW ---

    # Filter and Lemmatize for the main vocabulary list
    filtered_lemmas = set()
    for word in combined_source_vocab:
        if (word.isalpha() and word.isascii() and
            MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH and
            word not in english_stopwords and word not in all_names):
            
            lemma = lemmatizer.lemmatize(word.lower(), pos=wordnet.VERB)
            
            if (MIN_WORD_LENGTH <= len(lemma) <= MAX_WORD_LENGTH and lemma not in english_stopwords):
                filtered_lemmas.add(lemma)

    logging.info(f"Found {len(filtered_lemmas)} suitable, unique lemmas for the game.")
    return sorted(list(filtered_lemmas))

def rank_vocabulary(vocab_list):
    """Ranks the vocabulary list based on COCA word frequency."""
    logging.info(f"Step 3: Ranking vocabulary using top {COCA_WORDS_FOR_RANKING} from COCA...")
    coca_for_ranking = top_n_list('en', COCA_WORDS_FOR_RANKING)
    # Create a map of {word: rank} for fast lookups. Lower rank = more frequent.
    coca_rank_map = {word: i for i, word in enumerate(coca_for_ranking)}

    # Sort the list. Words not in the top 5000 get a rank of infinity and go to the end.
    sorted_vocab = sorted(vocab_list, key=lambda word: coca_rank_map.get(word, float('inf')))
    return sorted_vocab

def generate_embeddings_and_index(sorted_vocab, sbert_model):
    """Generates SBERT embeddings and builds the Annoy index for fast lookups."""
    logging.info(f"Step 4: Generating {VECTOR_DIMENSIONS}-dim vectors with batch size {SBERT_BATCH_SIZE}...")
    embeddings = sbert_model.encode(
        sorted_vocab,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=SBERT_BATCH_SIZE
    )
    word_vectors = {word: embeddings[i].astype(np.float16) for i, word in enumerate(sorted_vocab)}

    logging.info("Step 5: Building Annoy index and ID map...")
    id_to_word = {i: word for i, word in enumerate(word_vectors.keys())}
    annoy_index = AnnoyIndex(VECTOR_DIMENSIONS, 'angular')
    for i, word in id_to_word.items():
        annoy_index.add_item(i, word_vectors[word])
    annoy_index.build(100) # 100 trees is a good balance of accuracy and build time

    return word_vectors, annoy_index, id_to_word

def save_artifacts(sorted_vocab, word_vectors, annoy_index, id_to_word):
    """Saves all generated artifacts to disk."""
    logging.info("Step 6: Saving all data files...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save core assets
    joblib.dump(word_vectors, OUTPUT_VECTORS_PATH, compress=COMPRESSION_LEVEL, protocol=4)
    annoy_index.save(OUTPUT_ANNOY_INDEX_PATH)
    with open(OUTPUT_ID_TO_WORD_PATH, 'w') as f:
        json.dump(id_to_word, f, indent=2)

    # Save helper/game-specific files
    common_words_to_skip = sorted_vocab[:NUM_COMMON_WORDS_TO_SKIP]
    with open(OUTPUT_COMMON_WORDS_PATH, 'w') as f:
        json.dump(common_words_to_skip, f, indent=2)

    # Use dynamic percentiles to create difficulty buckets
    total_words = len(sorted_vocab)
    easy_split = int(total_words * EASY_PERCENTILE)
    medium_split = int(total_words * MEDIUM_PERCENTILE)

    final_word_buckets = {
        "easy": sorted_vocab[:easy_split],
        "medium": sorted_vocab[easy_split:medium_split],
        "hard": sorted_vocab[medium_split:]
    }
    with open(OUTPUT_VOCAB_PATH, 'w') as f:
        json.dump(final_word_buckets, f)
    
    logging.info(f"Saved {len(final_word_buckets['easy'])} easy, {len(final_word_buckets['medium'])} medium, and {len(final_word_buckets['hard'])} hard words.")

def run_precomputation_pipeline():
    """Executes the full precomputation pipeline from start to finish."""
    start_time = time.time()
    logging.info("="*50)
    logging.info("  Starting Enhanced SBERT Precomputation Pipeline")
    logging.info("="*50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_resources(device)
    initial_vocab = build_vocabulary(model)
    sorted_vocab = rank_vocabulary(initial_vocab)
    word_vectors, annoy_index, id_to_word = generate_embeddings_and_index(sorted_vocab, model)
    save_artifacts(sorted_vocab, word_vectors, annoy_index, id_to_word)

    end_time = time.time()
    logging.info("="*50)
    logging.info(f"✅ Final Precomputation complete in {end_time - start_time:.2f} seconds!")
    logging.info("="*50)

if __name__ == "__main__":
    run_precomputation_pipeline()