import logging
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Purpose of this script:
# - Convert words to numerical indices based on the vocabulary


def repr_numerical(text_sequences, word_index, max_sequence_length=None):
    logging.info("[representation] Converting words to numerical indices...")

    # Create Tokenizer
    #tokenizer = Tokenizer()

    # Fit on the entire text to ensure consistency with the vocabulary
    #tokenizer.fit_on_texts([' '.join(map(str, sequence)) for sequence in text_sequences])

    # Convert text sequences to numerical indices
    #numerical_indices = tokenizer.texts_to_sequences([' '.join(map(str, sequence)) for sequence in text_sequences])

    # Padding (optional, set max_sequence_length=None to disable)
    #if max_sequence_length:
    #    numerical_indices = pad_sequences(numerical_indices, maxlen=max_sequence_length, padding='pre')

    # Explicitly cast the NumPy array to a TensorFlow tensor
    #numerical_indices_tensor = tf.constant(numerical_indices, dtype=tf.int32)

    #return numerical_indices_tensor
    return 0