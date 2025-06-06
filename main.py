from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import random
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from functools import lru_cache

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


        