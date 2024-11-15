# MinHash
An efficient implementation of MinHash algorithm.

# Dataset
The English Wikipedia Dataset:
https://huggingface.co/datasets/lucadiliello/english_wikipedia

# Key Implementation Points

1. **Text Preprocessing and Tokenization**:
   - **Text is normalized**: Converted to lowercase, and punctuation is removed.
   - `nltk` is used to generated n-grams (default: 3-grams) from the token sequence, which is then converted to a set.

2. **Minhashing Process**:
   - **Hashing with xxhash**: The script uses `xxhash.xxh32` to compute 32-bit hash values for n-grams.
   - **Random Linear Transformations as Hashes**: For MinHashing, `NUM_HASHES` number of random linear transformations ($h_i(x) = a_i  x + b_i$) are generated. Since random linear transformations are *min-wise independent*, the coefficients $a_i$ and $b_i$ are random integers in the range `[1, 2^32-1]` and `[0, 2^32-1]` respectively.
   - **`np.uint32` Wrapping**: Hash values are wrapped using `np.uint32`, meaning the result is constrained to a 32-bit unsigned integer range (`[0, 2^32 - 1]`). This ensures that the hash values fit within 32 bits and can "wrap around" if they exceed this range, maintaining consistent as well as efficient computation.

3. **Jaccard Similarity Computation**:
   - **Approximate Jaccard Similarity (MinHash-based)**: MinHash signatures are compared by counting the number of matching hash values between the query and document MinHash signatures.
   - $$\text{Jaccard Similarity (approx.)} = \frac{\text{Number of matches in MinHash signatures}}{\text{Number of hashes}}.$$

4. **Storage of MinHash Signatures**:
   - MinHash signatures are computed and stored as `.npy` files in the `minhashes` directory. If a signature already exists for a document, it is loaded to avoid redundant computation.

5. **Document Relevance**:
   - For a given query document, its MinHash signature is compared with the signatures of other documents. Documents with a MinHash-based Jaccard similarity above the `JACCARD_THRESHOLD` are considered relevant.
