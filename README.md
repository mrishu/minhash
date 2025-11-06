# MinHash

An efficient implementation of MinHash algorithm.

## Dataset

The English Wikipedia Dataset:
<https://huggingface.co/datasets/lucadiliello/english_wikipedia>

## Dependencies

Main dependencies for the implementation:

```
pip install numpy nltk xxhash
```

To run on the dataset:

```
pip install datasets tqdm
```

## Key Implementation Points

1. **Text Preprocessing and Tokenization**:
   - **Text is normalized**: Converted to lowercase, and punctuation is removed.
   - `nltk` is used to generate n-grams (default: 3-grams) from the token sequence, which is then converted to a set.

2. **Minhashing Process**:
    - **Hashing with xxhash**: The script uses `xxhash.xxh32` to compute 32-bit hash values for n-grams.
    - **Random Linear Transformations as Hashes**: For MinHashing, we need a family of *min-wise independent* hashes. These are approximated by
        $$h_i(x) = a_i x + b_i mod p,$$
        where $a_i$ and $b_i$ are random integers and $p$ is a large prime. Linear transformations by themselves are not min-wise independent because they preserve or reverse order; the required randomness arises mainly from the modulus. In practice, we approximate this further using 32-bit machine truncation (equivalent to taking modulo $2^{32}$). Since multiplication followed by truncation with even integers reduces bit entropy, we restrict both $a_i$ and $b_i$ to **odd** integers. Thus we generate `NUM_HASHES` random linear transformations of the form
        $$h_i(x) = a_i x + b_i mod 2^{32},$$
    where $a_i$ is an odd integer in $[1, 2^{32} − 1]$ and $b_i$ is an odd integer in $[0, 2^{32} − 1]$. The output is stored as`np.uint32` for truncation.
    - **np.uint32 Wrapping**: Hash values are wrapped using `np.uint32`, constraining them to the 32-bit unsigned integer range $[0, 2^{32} − 1]$. This ensures consistent wrap-around behaviour and efficient computation.

3. **Jaccard Similarity Computation**:
   - **Approximate Jaccard Similarity (MinHash-based)**: MinHash signatures are compared by counting the number of matching hash values between the query and document MinHash signatures.
   - $$\text{Jaccard Similarity (approx.)} = \frac{\text{Number of matches in MinHash signatures}}{\text{Number of hashes}}.$$

4. **Storage of MinHash Signatures**:
   - MinHash signatures are computed and stored as `.npy` files in the `computed_minhashes` directory. If a signature already exists for a document, it is loaded to avoid redundant computation.

5. **Document Relevance**:
   - For a given query document, its MinHash signature is compared with the signatures of other documents. Documents with a MinHash-based Jaccard similarity above the `JACCARD_THRESHOLD` are considered relevant.
