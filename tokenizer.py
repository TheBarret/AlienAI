import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Purpose of this script:
# - Tokenize the text into individual words or subword tokens
# - It expects the data to be cleaned up by the preprocesser

def tk_load(text, max_sequence_length=None):
    logging.info("[tokenizer] Tokenizing data...")
   
    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
   
    # Get the total number of unique words
    total_words = len(tokenizer.word_index) + 1
    logging.info(f"[tokenizer] {total_words} unique words found")

    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences([text])[0]
    logging.info("[tokenizer] Sequence building...")

    # Padding (optional, set max_sequence_length=None to disable)
    if max_sequence_length:
        sequences = pad_sequences([sequences], maxlen=max_sequence_length, padding='pre')[0]

    return sequences, total_words