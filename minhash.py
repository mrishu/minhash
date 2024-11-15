# pip install datasets numpy nltk xxhash tqdm

import numpy as np
import nltk
import re
import xxhash
from datasets import load_dataset
from tqdm import tqdm
import os
from typing import Union

nltk.download("punkt")

# Filter docs with Jaccard similarity >= JACCARD_THRESHOLD (Change this according to need)
JACCARD_THRESHOLD = 0.2

# Can change this but delete precomputed minhashes `rm -rf minhashes`
NGRAMS_LEN = 3
NUM_HASHES = 100

# Dont change this as 32 bit hashes are being generated using xxhash.xxh32
MAX_HASH = np.uint32(2**32 - 1)


def fast_tokenize(text: str):
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = text.split()
    return tokens


class Minhasher:
    def __init__(self, num_hashes=NUM_HASHES, ngrams_len=NGRAMS_LEN):
        self.num_hashes = num_hashes
        self.ngrams_len = ngrams_len
        self.a, self.b = self._generate_lineartransform_hash_funcs()

    # NOTE: Random linear transformations are min-wise independent
    def _generate_lineartransform_hash_funcs(self) -> tuple[np.ndarray, np.ndarray]:
        """Generates fixed coefficients a's and b's for hash functions h_i(x) = a_i * x + b_i."""
        np.random.seed(101)
        a = (
            np.random.randint(MAX_HASH - 1, size=self.num_hashes, dtype=np.uint32) + 1
        )  # `a`s in range [1, MAX_HASH]
        b = np.random.randint(
            MAX_HASH, size=self.num_hashes, dtype=np.uint32
        )  # `b`s in range [0, MAX_HASH]
        # making the first linear transformation identity
        a[0] = 1
        b[0] = 0
        # verify that no two linear transformations are equal
        for i in range(self.num_hashes):
            for j in range(i + 1, self.num_hashes):
                if a[i] == a[j] and b[i] == b[j]:
                    raise ValueError("Two hash functions are equal! Retry!")
        return a, b

    def _compute_minhash(self, ngrams_set: set) -> np.ndarray:
        minhash_sig = np.full(self.num_hashes, MAX_HASH)
        for ngram in ngrams_set:
            ngram_str = "--".join(ngram).encode("utf-8")
            hash_val = np.uint32(xxhash.xxh32(ngram_str, seed=101).intdigest())
            hashes = self.a * hash_val + self.b
            minhash_sig = np.minimum(hashes, minhash_sig)
        return minhash_sig

    def minhash(self, text: str) -> Union[np.ndarray, None]:
        tokens = fast_tokenize(text)
        ngrams_set = set(nltk.ngrams(tokens, self.ngrams_len))
        # ngrams_set will be empty if there are less than ngrams_len words
        if not ngrams_set:
            return None
        minhash_sig = self._compute_minhash(ngrams_set)
        return minhash_sig


def jaccard_similarity_from_minhash(
    minhash_sig1: np.ndarray, minhash_sig2: np.ndarray
) -> float:
    assert len(minhash_sig1) == len(minhash_sig2), "Signatures must be of equal length."
    matching_hashes = np.sum(minhash_sig1 == minhash_sig2)
    jaccard_similarity = matching_hashes / len(minhash_sig1)
    return jaccard_similarity


def jaccard_similarity_exact(text1: str, text2: str, ngrams_len=3) -> float:
    tokens1 = fast_tokenize(text1)
    ngrams1 = set(nltk.ngrams(tokens1, ngrams_len))
    tokens2 = fast_tokenize(text2)
    ngrams2 = set(nltk.ngrams(tokens2, ngrams_len))
    return len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2))


ds = load_dataset("lucadiliello/english_wikipedia", split="train")

minhasher = Minhasher()

query = ds[0]["maintext"]  # using first document as query
# query = input("Enter query: ")
minhash_q = minhasher.minhash(query)
if minhash_q is None:
    raise ValueError("Query must contain at least one n-gram!")

print("Minhash for query generated!")

directory = "minhashes"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created. Minhashes will be saved here.")
else:
    print(f"Directory '{directory}' already exists. Minhashes will be saved here.")

relevant_docs_count = 0

for doc in tqdm(ds):
    text = doc["maintext"]
    filename = doc["filename"]  # filenames are numeric
    file_path = os.path.join(directory, filename + ".npy")

    # skip empty text
    if not text or not text.strip():
        continue

    # load or compute minhash
    if os.path.exists(file_path):
        minhash_doc = np.load(file_path)
    else:
        minhash_doc = minhasher.minhash(text)
        if minhash_doc is not None:
            np.save(file_path, minhash_doc)
        else:
            continue

    # compute jaccard similarity
    js_minhash = jaccard_similarity_from_minhash(minhash_q, minhash_doc)
    if js_minhash >= JACCARD_THRESHOLD:
        relevant_docs_count += 1
        print(f"Relevant Document Found: {doc['url']}")
        print(f"Jaccard Similarity (MinHash): {js_minhash:.2f}")

print(f"\nTotal relevant documents found: {relevant_docs_count}")
