from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Set
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import joblib
import psutil
from annoy import AnnoyIndex
import requests # <-- Import requests

app = FastAPI()

# --- Globals ---
WORD_VECTORS: Dict[str, np.ndarray] = {}
WORD_BUCKETS: Dict[str, List[str]] = {}
LEMMA_MAP: Dict[str, str] = {}
COMMON_WORDS_TO_SKIP: Set[str] = set()
ANTONYM_MAP: Dict[str, List[str]] = {}
ANNOY_INDEX: AnnoyIndex = None
ID_TO_WORD: Dict[int, str] = {}
VECTOR_DIMENSIONS = 300

# --- Gameplay Tuning Constants (YOUR TWEAKS INCORPORATED) ---
SIMILARITY_FLOOR, SIMILARITY_CEILING, PROGRESS_CURVE_POWER = 0.4, 0.8, 1.35
# Note: As you've set, antonyms will now receive a very high score.
ANTONYM_PROGRESS_SCORE = 90
HINT_IDEAL_MIN, HINT_IDEAL_MAX = 0.67, 0.9

@app.on_event("startup")
def load_precomputed_data():
    print("Verifying and loading precomputed data...")
    
    # =====================================================================
    # NEW: Self-Healing Data Downloader
    # This block ensures all required data files are present before loading.
    # =====================================================================
    DATA_DIR = "data"
    BASE_URL = "https://raw.githubusercontent.com/danielphingston/word-similarity-server/master/data/"
    
    # List of all required data files
    required_files = [
        "word_vectors.joblib", "game_vocabulary.json", "lemma_map.joblib",
        "common_words.json", "antonym_map.joblib", "word_vectors.ann", "id_to_word.json"
    ]

    os.makedirs(DATA_DIR, exist_ok=True)

    for filename in required_files:
        local_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(local_path):
            url = f"{BASE_URL}{filename}"
            print(f"Data file '{filename}' not found. Downloading from GitHub...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded '{filename}'.")
            except requests.exceptions.RequestException as e:
                # If a file can't be downloaded, the app can't start.
                raise RuntimeError(f"FATAL: Could not download essential data file '{filename}'. Error: {e}") from e

    # --- Load the data now that we know it exists ---
    paths = {
        "vectors": "data/word_vectors.joblib", "vocab": "data/game_vocabulary.json",
        "lemmas": "data/lemma_map.joblib", "common": "data/common_words.json",
        "antonyms": "data/antonym_map.joblib", "annoy_index": "data/word_vectors.ann",
        "id_map": "data/id_to_word.json"
    }
    
    global WORD_VECTORS, WORD_BUCKETS, LEMMA_MAP, COMMON_WORDS_TO_SKIP, ANTONYM_MAP, ANNOY_INDEX, ID_TO_WORD
    WORD_VECTORS = joblib.load(paths["vectors"])
    LEMMA_MAP = joblib.load(paths["lemmas"])
    ANTONYM_MAP = joblib.load(paths["antonyms"])
    with open(paths["vocab"], "r") as f: WORD_BUCKETS = json.load(f)
    with open(paths["common"], "r") as f: COMMON_WORDS_TO_SKIP = set(json.load(f))
    with open(paths["id_map"], "r") as f: ID_TO_WORD = {int(k): v for k, v in json.load(f).items()}
    
    ANNOY_INDEX = AnnoyIndex(VECTOR_DIMENSIONS, 'angular')
    ANNOY_INDEX.load(paths["annoy_index"]) 
        
    print(f"✅ Loaded {len(WORD_VECTORS)} vectors, {ANNOY_INDEX.get_n_items()} Annoy items, and {len(ANTONYM_MAP)} antonym entries.")
    process = psutil.Process(os.getpid()); memory_mb = process.memory_info().rss / (1024*1024)
    print(f"🧠 Post-startup memory usage: {memory_mb:.2f} MB")

def lemmatize(word: str): return LEMMA_MAP.get(word.lower().strip(), word.lower().strip())
def similarity_score(w1: str, w2: str): return float((cosine_similarity([WORD_VECTORS[w1]], [WORD_VECTORS[w2]])[0][0] + 1) / 2)
def get_progress_score(s: float):
    if s < SIMILARITY_FLOOR: return 0
    if s >= SIMILARITY_CEILING: return 100
    norm_s = (s - SIMILARITY_FLOOR) / (SIMILARITY_CEILING - SIMILARITY_FLOOR)
    return int((norm_s ** PROGRESS_CURVE_POWER) * 100)

@app.get("/similarity")
def get_similarity(word1: str = Query(...), word2: str = Query(...)):
    if len(word1) < 3 or len(word2) < 3: return {"similarity": -1, "progress_score": 0, "isValidGuess": False, "reason": "Words must be at least 3 characters long."}
    lemma1, lemma2 = lemmatize(word1), lemmatize(word2)
    if lemma1 not in WORD_VECTORS: return {"similarity": -1, "progress_score": 0, "isValidGuess": False, "reason": f"Your guess '{word1}' is not a valid word in the game."}
    if lemma2 not in WORD_VECTORS: return {"similarity": -1, "progress_score": 0, "isValidGuess": False, "reason": f"The target word '{word2}' is not valid."}
    if lemma1 == lemma2: return {"similarity": 1.0, "progress_score": 100, "isValidGuess": True, "reason": "Perfect match!"}
    raw_sim = similarity_score(lemma1, lemma2)
    if lemma1 in ANTONYM_MAP.get(lemma2, []): return {"similarity": raw_sim, "progress_score": ANTONYM_PROGRESS_SCORE, "isValidGuess": True, "reason": "It's an antonym! That's the opposite of what you want."}
    return {"similarity": raw_sim, "progress_score": min(get_progress_score(raw_sim), 99), "isValidGuess": True, "reason": "Valid guess."}

class HintRequest(BaseModel):
    word: str
    exclude: List[str] = []

@app.post("/hint")
def get_hint(data: HintRequest):
    target_word_lemma = lemmatize(data.word)
    if target_word_lemma not in WORD_VECTORS:
        return {"hint": None, "reason": "word not in model"}

    exclude_lemmas = {lemmatize(w) for w in data.exclude}
    exclude_lemmas.add(target_word_lemma)

    target_vector = WORD_VECTORS[target_word_lemma]
    neighbor_ids = ANNOY_INDEX.get_nns_by_vector(target_vector, 200)

    ideal_candidates = []
    all_candidates = []
    
    for item_id in neighbor_ids:
        hint_word = ID_TO_WORD.get(item_id)
        if hint_word and hint_word not in exclude_lemmas:
            # CORRECTED: Reuse the exact same scoring functions
            raw_sim = similarity_score(target_word_lemma, hint_word)
            progress = get_progress_score(raw_sim)

            hint_data = {
                "hint": hint_word,
                "similarity": raw_sim,
                "progress_score": min(progress, 99)
            }
            all_candidates.append(hint_data)
            
            # Use the server-side constants for the ideal hint range
            if HINT_IDEAL_MIN <= raw_sim <= HINT_IDEAL_MAX:
                ideal_candidates.append(hint_data)

    if ideal_candidates:
        return random.choice(ideal_candidates)
    
    if all_candidates:
        return all_candidates[0]

    return {"hint": None, "reason": "no usable hint found"}

@app.get("/")
def read_root(): return {"message": "Welcome to the Word Similarity API!"}

@app.get("/random-word")
def get_random_word(difficulty: str = Query("easy", regex="^(easy|medium|hard)$")):
    words = WORD_BUCKETS.get(difficulty, [])
    if not words: return {"word": None, "error": f"No words found for difficulty '{difficulty}'"}
    eligible_words = [w for w in words if len(w) >= 4 and w not in COMMON_WORDS_TO_SKIP]
    if not eligible_words: return {"word": None, "error": f"No eligible words found for difficulty '{difficulty}' after filtering."}
    return {"word": random.choice(eligible_words), "difficulty": difficulty}