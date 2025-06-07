from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk
import json
import os
import joblib

# ================================
# Load Precomputed Data & NLP
# ================================
print("Downloading NLTK WordNet...")
nltk.download('wordnet')

app = FastAPI()
lemmatizer = WordNetLemmatizer()

WORD_VECTORS: Dict[str, np.ndarray] = {}
WORD_BUCKETS: Dict[str, List[str]] = {}

# Constants for the new hint logic
HINT_TARGET_MIN = 0.4  # Corresponds to a score of 40
HINT_TARGET_MAX = 0.6  # Corresponds to a score of 60

@app.on_event("startup")
def load_precomputed_data():
    print("Loading precomputed word vectors and vocabulary...")
    vectors_path = "data/word_vectors.joblib"
    vocab_path = "data/game_vocabulary.json"

    if not os.path.exists(vectors_path) or not os.path.exists(vocab_path):
        print(f"ERROR: Precomputed data not found! Please run 'precompute.py' first.")
        return

    global WORD_VECTORS
    WORD_VECTORS = joblib.load(vectors_path)
    
    with open(vocab_path, "r") as f:
        global WORD_BUCKETS
        WORD_BUCKETS = json.load(f)
        
    print(f"✅ Successfully loaded {len(WORD_VECTORS)} vectors and {sum(len(v) for v in WORD_BUCKETS.values())} words.")

# ================================
# Utils
# ================================
def lemmatize(word: str) -> str:
    return lemmatizer.lemmatize(word.lower().strip(), pos='n')

def similarity_score(w1: str, w2: str) -> float | None:
    v1 = WORD_VECTORS.get(w1)
    v2 = WORD_VECTORS.get(w2)
    if v1 is None or v2 is None:
        return None
    raw_similarity = cosine_similarity([v1], [v2])[0][0]
    return float((raw_similarity + 1) / 2)

# ================================
# Endpoints
# ================================
@app.get("/")
def read_root():
    return {"message": "Welcome to the Word Similarity API!"}

@app.get("/similarity")
def get_similarity(word1: str = Query(...), word2: str = Query(...)):
    lemma1 = lemmatize(word1)
    lemma2 = lemmatize(word2)
    sim = similarity_score(lemma1, lemma2)
    return {
        "word1": lemma1,
        "word2": lemma2,
        "similarity": sim if sim is not None else -1
    }

@app.get("/random-word")
def get_random_word(difficulty: str = Query("easy", regex="^(easy|medium|hard)$")):
    words = WORD_BUCKETS.get(difficulty, [])
    if not words:
        return {"word": None, "error": f"No words found for difficulty '{difficulty}'"}
    word = random.choice(words)
    return {"word": word, "difficulty": difficulty}

class HintRequest(BaseModel):
    word: str
    exclude: List[str] = []
    threshold: float = 0.7
    tolerance: float = 0.05

@app.post("/hint")
def get_hint(data: HintRequest):
    target_word_lemma = lemmatize(data.word)
    
    if target_word_lemma not in WORD_VECTORS:
        return {"hint": None, "reason": "word not in model"}

    exclude_lemmas = {lemmatize(w) for w in data.exclude}
    exclude_lemmas.add(target_word_lemma)

    # =====================================================================
    # NEW: Sophisticated Hint Logic
    # =====================================================================
    
    all_candidates = []
    ideal_candidates = []

    # 1. Iterate through the vocabulary and calculate scores for all possible hints
    for vocab_word in WORD_VECTORS:
        if vocab_word not in exclude_lemmas:
            score = similarity_score(target_word_lemma, vocab_word)
            if score is not None:
                hint_data = {"hint": vocab_word, "similarity": score}
                all_candidates.append(hint_data)
                
                # 2. Check if the candidate falls into our "ideal" hint range
                if HINT_TARGET_MIN <= score <= HINT_TARGET_MAX:
                    ideal_candidates.append(hint_data)

    # 3. If we found any ideal candidates, pick one randomly.
    if ideal_candidates:
        print(f"Found {len(ideal_candidates)} ideal hints. Picking one randomly.")
        return random.choice(ideal_candidates)
        
    # 4. If no ideal hints were found, fall back to the old logic.
    #    Sort all candidates by score and return the best one available.
    if all_candidates:
        print("No ideal hints found. Falling back to providing the best possible hint.")
        all_candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return all_candidates[0]
        
    # 5. If for some reason no hints could be generated at all.
    return {"hint": None, "reason": "no usable hint found"}