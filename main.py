from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import random
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from functools import lru_cache
import json
from nltk.stem import WordNetLemmatizer
import nltk

# ================================
# Load NLP & Embeddings
# ================================

print("Downloading NLTK WordNet...")
nltk.download('wordnet')

print("Loading GloVe Twitter embeddings...")
model = api.load("glove-twitter-25")

app = FastAPI()

print("Loading word difficulty buckets...")
with open("words.json", "r") as f:
    word_buckets = json.load(f)

# ================================
# Utils
# ================================

lemmatizer = WordNetLemmatizer()

def lemmatize(word: str) -> str:
    word = word.lower()
    lemma = lemmatizer.lemmatize(word, pos='n')  # noun-only lemmatization
    return lemma

def get_vector(word: str):
    try:
        return model[word]
    except KeyError:
        return None

def similarity_score(w1: str, w2: str):
    v1 = get_vector(w1)
    v2 = get_vector(w2)
    if v1 is None or v2 is None:
        return None
    return float((cosine_similarity([v1], [v2])[0][0]+1)/2)

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
    words = word_buckets.get(difficulty, [])
    if not words:
        return {"word": None, "error": f"No words found for difficulty '{difficulty}'"}
    word = random.choice(words)
    return {"word": word, "difficulty": difficulty}


class HintRequest(BaseModel):
    word: str
    exclude: List[str] = []
    threshold: float = 0.7
    tolerance: float = 0.05  # Optional: how close to threshold

@app.post("/hint")
def get_hint(data: HintRequest):
    try:
        similar_words = model.most_similar(data.word, topn=50)
    except KeyError:
        return {"hint": None, "reason": "word not in model"}

    exclude_set = set(data.exclude)
    exclude_set.add(data.word)

    # Reverse sort to get least similar last
    for word, score in reversed(similar_words):
        if word not in exclude_set:
            return {"hint": word, "similarity": score}

    return {"hint": None, "reason": "no usable hint found"}


        