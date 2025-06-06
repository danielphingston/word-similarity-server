import json
import re
import random
from tqdm import tqdm
from nltk.corpus import wordnet as wn, names
from wordfreq import word_frequency
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('wordnet')
nltk.download('names')

# ========================
# Config
# ========================
MAX_WORDS_PER_BUCKET = 2000  # Set to None to include all
EASY_FREQ_MIN = 1e-4
MEDIUM_FREQ_MIN = 1e-5
MIN_WORD_LENGTH = 4

# ========================
# Helpers
# ========================
lemmatizer = WordNetLemmatizer()

def lemmatize(word: str) -> str:
    return lemmatizer.lemmatize(word.lower(), pos='n')  # noun lemmatization only

def is_clean_english(word):
    return word.isalpha() and word.islower() and re.match("^[a-z]+$", word)

# ========================
# Step 1: Load Only Noun Lemmas from WordNet
# ========================
print("Gathering noun lemmas from WordNet...")
lemmas = set()
for synset in tqdm(list(wn.all_synsets(pos='n'))):  # Only nouns
    for lemma in synset.lemmas():
        word = lemma.name().lower()
        lemmas.add(word)

print(f"Total noun lemmas from WordNet: {len(lemmas)}")

# ========================
# Step 2: Remove Names
# ========================
print("Removing names using nltk.corpus.names...")
name_set = set(n.lower() for n in names.words())
lemmas = [w for w in lemmas if w not in name_set]

# ========================
# Step 3: Filter Words and Get Frequencies
# ========================
print("Filtering and computing word frequencies...")
words_and_freqs = []
for word in tqdm(lemmas):
    if not is_clean_english(word):
        continue
    if len(word) < MIN_WORD_LENGTH:
        continue
    freq = word_frequency(word, 'en', wordlist='large')
    if freq == 0:
        continue
    words_and_freqs.append((word, freq))

print(f"Usable noun words: {len(words_and_freqs)}")

# ========================
# Step 4: Split by Frequency
# ========================
words_and_freqs.sort(key=lambda x: x[1], reverse=True)

easy, medium, hard = [], [], []

for word, freq in words_and_freqs:
    if freq >= EASY_FREQ_MIN:
        easy.append(word)
    elif freq >= MEDIUM_FREQ_MIN:
        medium.append(word)
    else:
        hard.append(word)

random.shuffle(easy)
random.shuffle(medium)
random.shuffle(hard)

if MAX_WORDS_PER_BUCKET:
    easy = easy[:MAX_WORDS_PER_BUCKET]
    medium = medium[:MAX_WORDS_PER_BUCKET]
    hard = hard[:MAX_WORDS_PER_BUCKET]

# ========================
# Step 5: Save JSON
# ========================
data = {"easy": easy, "medium": medium, "hard": hard}
with open("noun_difficulty_words.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ Done. Saved 'noun_difficulty_words.json'")
print(f"  easy: {len(easy)}")
print(f"  medium: {len(medium)}")
print(f"  hard: {len(hard)}")
