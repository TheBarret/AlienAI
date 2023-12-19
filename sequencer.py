import logging
import numpy as np

# Purpose of this script:
# - Generate input-output pairs for training
# - For text generation, each input should be a sequence of words,
#   and the corresponding output should be the next word in the sequence

def seq_pairs(text_sequences):
    logging.info("[sequencer] Generating IO pairs...")
    # Generate input-output pairs
    input_sequences, output_words = [], []
    for i in range(1, len(text_sequences)):
        n_gram_sequence = text_sequences[:i+1]
        input_sequences.append(n_gram_sequence[:-1])
        output_words.append(n_gram_sequence[-1])
    
    # Convert to numpy arrays with dtype=object
    logging.info("[sequencer] Converting to numpy arrays ...")
    input_sequences, output_words = np.array(input_sequences, dtype=object), np.array(output_words, dtype=object)
    input_sequences, output_words = np.array(input_sequences), np.array(output_words)

    return input_sequences, output_words