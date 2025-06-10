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
from enum import Enum

app = FastAPI()

# --- Globals (Simplified) ---
WORD_VECTORS: Dict[str, np.ndarray] = {}
WORD_BUCKETS: Dict[str, List[str]] = {}
COMMON_WORDS_TO_SKIP: Set[str] = set()
ANNOY_INDEX: AnnoyIndex = None
ID_TO_WORD: Dict[int, str] = {}
VECTOR_DIMENSIONS = 768
LEMMATIZATION_MAP: Dict[str, str] = {}

# =====================================================================
# FINAL SBERT GAMEPLAY TUNING CONSTANTS
# =====================================================================
SIMILARITY_FLOOR = 0.75
SIMILARITY_CEILING = 0.93
# PROGRESS_CURVE_POWER = .67

# --- Parameters for each curve type ---
GENEROUS_POWER = 0.67
CHALLENGING_POWER = 1.5
# S_CURVE_STEEPNESS controls how quickly the score accelerates in the middle.
# Higher values make the jump from ~10 to ~90 points happen over a smaller similarity range.
S_CURVE_STEEPNESS = 10.0

HINT_IDEAL_MIN, HINT_IDEAL_MAX = 0.80, 0.90

class CurveType(Enum):
    GENEROUS = 1    # Fast start, slow finish. Good for encouragement.
    LINEAR = 2      # Proportional progress. Fair and predictable.
    CHALLENGING = 3 # Slow start, fast finish. Rewards precision.
    S_CURVE = 4     # Natural feel: slow, then fast, then slow. Highly tunable.

@app.on_event("startup")
def load_precomputed_data():
    print("Loading precomputed SBERT data...")
    paths = {
        "vectors": "data/word_vectors.joblib", "vocab": "data/game_vocabulary.json",
        "common": "data/common_words.json", "annoy_index": "data/word_vectors.ann",
        "id_map": "data/id_to_word.json",
        "lemma_map": "data/lemmatization_map.json"
    }
    if not all(os.path.exists(p) for p in paths.values()):
        print("ERROR: Precomputed data not found! Please run 'precompute.py' first.")
        return

    global WORD_VECTORS, WORD_BUCKETS, COMMON_WORDS_TO_SKIP, ANNOY_INDEX, ID_TO_WORD, LEMMATIZATION_MAP
    WORD_VECTORS = joblib.load(paths["vectors"])
    with open(paths["vocab"], "r") as f: WORD_BUCKETS = json.load(f)
    with open(paths["common"], "r") as f: COMMON_WORDS_TO_SKIP = set(json.load(f))
    with open(paths["id_map"], "r") as f: ID_TO_WORD = {int(k): v for k, v in json.load(f).items()}
    with open(paths["lemma_map"], "r") as f: LEMMATIZATION_MAP = json.load(f) # <--- LOAD THE MAP

    ANNOY_INDEX = AnnoyIndex(VECTOR_DIMENSIONS, 'angular')
    ANNOY_INDEX.load(paths["annoy_index"])

    print(f"✅ Loaded {len(WORD_VECTORS)} vectors and {ANNOY_INDEX.get_n_items()} Annoy items.")
    print(f"✅ Loaded lemmatization map with {len(LEMMATIZATION_MAP)} entries.") # <--- NEW LOG
    process = psutil.Process(os.getpid()); memory_mb = process.memory_info().rss / (1024*1024)
    print(f"🧠 Post-startup memory usage: {memory_mb:.2f} MB")

def get_lemma(word: str) -> str:
    """Lemmatizes a word using the pre-computed map. Returns original word if not in map."""
    return LEMMATIZATION_MAP.get(word, word)

def similarity_score(w1: str, w2: str) -> float | None:
    v1 = WORD_VECTORS.get(w1)
    v2 = WORD_VECTORS.get(w2)
    if v1 is None or v2 is None: return None
    return float((cosine_similarity([v1], [v2])[0][0] + 1) / 2)

def get_progress_score(
    similarity: float,
    curve_type: CurveType = CurveType.S_CURVE # Default to the most versatile curve
) -> int:
    """
    Calculates a gameplay score from 0-100 based on a raw similarity value.
    Supports multiple curve shapes for different gameplay feels.
    """
    # 1. Handle edge cases
    if similarity < SIMILARITY_FLOOR:
        return 0
    if similarity >= SIMILARITY_CEILING:
        return 100

    # 2. Normalize the similarity to a 0.0 to 1.0 scale
    # This represents the player's progress within the active scoring range.
    norm_s = (similarity - SIMILARITY_FLOOR) / (SIMILARITY_CEILING - SIMILARITY_FLOOR)

    # 3. Apply the selected curve
    score_float = 0.0
    if curve_type == CurveType.GENEROUS:
        # y = x^0.67 -> Quick initial rise
        score_float = norm_s ** GENEROUS_POWER
    
    elif curve_type == CurveType.LINEAR:
        # y = x -> Straight line
        score_float = norm_s
        
    elif curve_type == CurveType.CHALLENGING:
        # y = x^1.5 -> Slow initial rise
        score_float = norm_s ** CHALLENGING_POWER
        
    elif curve_type == CurveType.S_CURVE:
        # Use a logistic function for a natural "S" shape.
        # It's slow at the ends (0-10 and 90-100) and fast in the middle.
        # We remap norm_s from [0, 1] to a range like [-5, 5] for the sigmoid.
        remapped_s = (norm_s - 0.5) * S_CURVE_STEEPNESS
        score_float = 1 / (1 + np.exp(-remapped_s))

    return int(score_float * 100)

@app.get("/similarity")
def get_similarity(word1: str = Query(...), word2: str = Query(...)):
    # 1. Sanitize and lemmatize user input
    player_guess_raw = word1.strip().lower()
    player_guess_lemma = get_lemma(player_guess_raw)
    
    # The secret word is assumed to be a lemma already from our system
    secret_word_lemma = word2.strip().lower()
    
    # 2. Basic validation on the raw input
    if len(player_guess_raw) < 3:
        return {
            "similarity": -1, "progress_score": 0, "isValidGuess": False,
            "reason": "Words must be at least 3 characters long."
        }

    # 3. Check if the *lemmatized* guess is a valid game word
    if player_guess_lemma not in WORD_VECTORS:
        return {
            "similarity": -1, "progress_score": 0, "isValidGuess": False,
            "reason": f"Your guess '{player_guess_raw}' is not a valid word in the game."
        }
    
    # This check is a safeguard, but typically the secret word should always be valid
    if secret_word_lemma not in WORD_VECTORS:
        return {
            "similarity": -1, "progress_score": 0, "isValidGuess": False,
            "reason": f"The target word '{secret_word_lemma}' is not valid."
        }
    
    # 4. Handle perfect match
    if player_guess_lemma == secret_word_lemma:
        return {
            "similarity": 1.0, "progress_score": 100, "isValidGuess": True,
            "reason": "Perfect match!", "lemmatized_guess": player_guess_lemma
        }
    
    # 5. Calculate similarity using the lemmatized versions
    raw_sim = similarity_score(player_guess_lemma, secret_word_lemma)
    progress = get_progress_score(raw_sim)

    # 6. Return the result. The user's guess is valid and was processed.
    return {
        "similarity": raw_sim, "progress_score": min(progress, 99), "isValidGuess": True,
        "reason": "Valid guess.", "lemmatized_guess": player_guess_lemma
    }
# ===========================================================

class HintRequest(BaseModel):
    word: str
    exclude: List[str] = []

@app.post("/hint")
def get_hint(data: HintRequest):
    target_word = data.word.strip()
    if target_word not in WORD_VECTORS:
        return {"hint": None, "reason": "word not in model"}

    exclude_words = {w.strip() for w in data.exclude}
    exclude_words.add(target_word)

    target_vector = WORD_VECTORS[target_word]
    neighbor_ids = ANNOY_INDEX.get_nns_by_vector(target_vector, 200)

    ideal_candidates, all_candidates = [], []
    for item_id in neighbor_ids:
        hint_word = ID_TO_WORD.get(item_id)
        if hint_word and hint_word not in exclude_words:
            raw_sim = similarity_score(target_word, hint_word)
            progress = get_progress_score(raw_sim)
            hint_data = {"hint": hint_word, "similarity": raw_sim, "progress_score": min(progress, 99)}
            all_candidates.append(hint_data)
            if HINT_IDEAL_MIN <= raw_sim <= HINT_IDEAL_MAX:
                ideal_candidates.append(hint_data)

    if ideal_candidates: return random.choice(ideal_candidates)
    if all_candidates: return all_candidates[0]
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