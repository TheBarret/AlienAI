import logging
import tensorflow as tf
from tensorflow.keras import layers, models

tf.get_logger().setLevel('ERROR')

# Purpose of this script:
# - Build a simple neural network model

def build_model(max_sequence_length, vocab_size):
    logging.info("[model] Building the neural network model...")

    model = models.Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_sequence_length))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.save('model.h5')
    
    logging.info("[model] model created successfully")
    model.summary()
    

    return model