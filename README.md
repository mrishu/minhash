# MinHash
An efficient implementation of MinHash algorithm.

# Dataset
The English Wikipedia Dataset:
https://huggingface.co/datasets/lucadiliello/english_wikipedia

# Key Implementation Points
1. Used `xxhash.xxh32` for efficient 32-bit hashing.
2. Generated `num_hashes` number of random linear transformations for simulating different hashes (as random linear transformations are min-wise independent).
