import logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Import all required stages
from preprocessing import (pp_load)
from tokenizer import (tk_load)
from sequencer import (seq_pairs)
from padding import (pad_uniform)
from vocabulary import (voc_build)
from representation import (repr_numerical)
from dataset import (dataset_create)
from dataset import (FEATURES)
from model import (build_model)

# Configuration
MAX_SEQUENCE_LENGTH = 100
BATCH_SIZE          = 32
SHUFFLE             = False
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
# eof configuration

# Phase: Dataset loading & cleaning
DATASET = pp_load(r'aliensim_dataset.txt')

# Phase: Tokenization
SEQUENCE, TOTAL_WORDS = tk_load(DATASET, MAX_SEQUENCE_LENGTH)

# Phase: Sequence generation
INPUT_SEQUENCE, OUTPUT_SEQUENCE = seq_pairs(SEQUENCE)

# Phase: Padding
PADDED_INPUT_SEQUENCE = pad_uniform(INPUT_SEQUENCE, MAX_SEQUENCE_LENGTH)

# Phase: Build vocabulary
WORD_INDEX = voc_build(SEQUENCE)

# Phase: Numerical representation
NUMERICAL_INPUT_SEQUENCE = repr_numerical(PADDED_INPUT_SEQUENCE, WORD_INDEX, MAX_SEQUENCE_LENGTH)

# Phase: Dataset creation
DATASET_TF = dataset_create(NUMERICAL_INPUT_SEQUENCE, OUTPUT_SEQUENCE, BATCH_SIZE, SHUFFLE)

# Phase: Build the model
MODEL = build_model(MAX_SEQUENCE_LENGTH, len(WORD_INDEX) + 1)
