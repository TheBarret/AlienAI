import logging
import tensorflow as tf

from model import build_model
from dataset import dataset_create

tf.get_logger().setLevel('ERROR')

# Purpose of this script:
# - Train the neural network model

# Set your actual values for these constants
MAX_SEQUENCE_LENGTH = 100
BATCH_SIZE          = 32
SHUFFLE             = True

# Assuming you have NUMERICAL_INPUT_SEQUENCE and OUTPUT_SEQUENCE from the previous steps
DATASET_TF = dataset_create(NUMERICAL_INPUT_SEQUENCE, OUTPUT_SEQUENCE, BATCH_SIZE, SHUFFLE)

# Build the model
VOCAB_SIZE = len(WORD_INDEX) + 1
model = build_model(MAX_SEQUENCE_LENGTH, VOCAB_SIZE)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
logging.info("[training] Training the neural network model...")
model.fit(DATASET_TF, epochs=5)

logging.info("[training] Neural network model training completed.")