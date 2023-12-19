import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Purpose of this script:
# - Pad sequences to ensure uniform length

def pad_uniform(sequences, max_sequence_length):
    logging.info("[padding] Padding uniform length...")
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
    return padded_sequences
