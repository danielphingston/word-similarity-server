from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import random
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy

# ================================
# Load NLP & Embeddings
# ================================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
model = api.load("glove-twitter-25")
all_words = list(model.key_to_index.keys())

app = FastAPI()

# ================================
# Utils
# ================================
def lemmatize(word: str) -> str:
    doc = nlp(word)
    return doc[0].lemma_ if doc else word

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
    return float(cosine_similarity([v1], [v2])[0][0])

def get_random_lemmatized_word():
    while True:
        word = random.choice(all_words)
        lemma = lemmatize(word)
        if get_vector(lemma) is not None:
            return lemma

# ================================
# Endpoints
# ================================
@app.get("/similarity")
def get_similarity(word1: str = Query(...), word2: str = Query(...)):
    lemma1 = lemmatize(word1)
    lemma2 = lemmatize(word2)
    sim = similarity_score(lemma1, lemma2)
    return {
        "word1": lemma1,
        "word2": lemma2,
        "similarity": sim if sim is not None else "unknown"
    }

@app.get("/random-word")
def get_random_word():
    word = get_random_lemmatized_word()
    return {"word": word}

class HintRequest(BaseModel):
    word: str
    exclude: List[str] = []
    threshold: float = 0.7
    tolerance: float = 0.05  # Optional: how close to threshold

@app.post("/hint")
def get_hint(data: HintRequest):
    base = lemmatize(data.word)
    exclude_set = {lemmatize(w) for w in data.exclude}
    candidates = []

    for candidate in all_words:
        lemma = lemmatize(candidate)
        if lemma in exclude_set or lemma == base:
            continue
        sim = similarity_score(base, lemma)
        if sim is None:
            continue
        if abs(sim - data.threshold) <= data.tolerance:
            candidates.append((lemma, sim))

    if not candidates:
        return {"hint": None, "reason": "no match near threshold"}

    # Return one with closest similarity
    candidates.sort(key=lambda x: abs(x[1] - data.threshold))
    best = candidates[0]
    return {"hint": best[0], "similarity": best[1]}
