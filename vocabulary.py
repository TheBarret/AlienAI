import logging
import numpy as np

# Purpose of this script:
# - Build a vocabulary of unique words or subword tokens from the training data

def voc_build(tokenized_sequences):
    logging.info("[vocabulary] Building word-to-index mapping...")
    flattened_sequences = []
    for seq in tokenized_sequences:
        for item in np.nditer(seq):
            flattened_sequences.append(str(item))
    # Create a set of unique words from tokenized sequences
    vocabulary = set(flattened_sequences)
    # Create a word-to-index mapping
    word_index = {word: index + 1 for index, word in enumerate(vocabulary)}
    return word_index