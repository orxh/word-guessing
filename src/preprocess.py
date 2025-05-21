"""
Script to preprocess the GloVe embeddings.
"""

import tqdm
import sklearn.feature_extraction.text as sklearn_text

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")

GLOVE_INFILE = "data/glove.42B.300d.txt"
GLOVE_OUTFILE = "data/glove.42B.300d.clean.txt"

MIN_LEN = 3
MAX_LEN = 15


def preprocess_glove():
    count = 0
    english_lemmas = {
        lemma.replace("_", " ")
        for syn in wn.all_synsets()
        for lemma in syn.lemma_names()
    }

    with open(GLOVE_INFILE, "r", encoding="utf8") as fi, open(
        GLOVE_OUTFILE, "w", encoding="utf8"
    ) as fo:
        total_lines = sum(1 for _ in fi)
        fi.seek(0)

        for line in tqdm.tqdm(fi, total=total_lines, desc="Processing GloVe lines"):
            word, vec = line.split(" ", 1)

            if not word.islower():
                continue

            word = word.strip()

            if len(word) < MIN_LEN or len(word) > MAX_LEN:
                continue

            if not word.isalpha():
                continue

            if word not in english_lemmas or word in sklearn_text.ENGLISH_STOP_WORDS:
                continue

            count += 1
            updated_line = f"{word} {vec}"
            fo.write(updated_line)

    logger.info("Wrote %d GloVe words", count)


if __name__ == "__main__":
    logger.info("Starting preprocessing...")

    logger.info("Preprocessing GloVe embeddings...")
    preprocess_glove()

    logger.info("Preprocessing complete!")
