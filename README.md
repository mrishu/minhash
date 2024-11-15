# MinHash
An efficient implementation of MinHash algorithm.

# Dataset
The English Wikipedia Dataset:
https://huggingface.co/datasets/lucadiliello/english_wikipedia

# Key Implementation Points

1. **Text Preprocessing and Tokenization**:
   - Text is tokenized using `nltk`, converted to lowercase, and punctuation is removed. N-grams (default: 3-grams) are generated from the tokens to represent the text.

2. **Minhashing Process**:
   - **Hashing with xxhash**: The script uses `xxhash.xxh32` to compute 32-bit hash values for n-grams. `xxhash.xxh32` is fast and produces high-quality hash values, with a seed value for randomness.
   - **Random Linear Transformations**: For MinHashing, random linear transformations (`h_i(x) = a_i * x + b_i`) are generated. The coefficients `a_i` and `b_i` are random integers, ensuring that hash functions are min-wise independent, which is key to MinHash’s accuracy in approximating Jaccard similarity.
   - **np.uint32 Wrapping**: Hash values are wrapped using `np.uint32`, meaning the result is constrained to a 32-bit unsigned integer range (0 to `2^32 - 1`). This ensures that the hash values fit within 32 bits and can "wrap around" if they exceed this range, maintaining consistent and efficient computation.

3. **Jaccard Similarity Computation**:
   - **Exact Jaccard Similarity**: The exact Jaccard similarity is computed as the ratio of the intersection to the union of the n-grams from two texts.
   - **Approximate Jaccard Similarity (MinHash-based)**: MinHash signatures are compared by counting the number of matching hash values between the query and document signatures. A higher number of matches indicates higher similarity.

4. **Storage of MinHash Signatures**:
   - MinHash signatures are computed and stored as `.npy` files in the `minhashes` directory. If a signature already exists for a document, it is loaded to avoid redundant computation.

5. **Document Relevance**:
   - For a given query document, its MinHash signature is compared with the signatures of other documents. Documents with a MinHash-based Jaccard similarity above the `JACCARD_THRESHOLD` are considered relevant.
