import logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Purpose of this script:
# - Create TensorFlow datasets from numerical representations

# numerical_indices
# output_sequence_reshaped

FEATURES = {
    "numerical_indices": "",
    "output_sequence": "" 
}

def dataset_create(numerical_indices, output_sequence, batch_size=32, shuffle=True):
    logging.info("[datasets] Creating TensorFlow datasets...")

    # Convert numerical indices and output sequence to TensorFlow tensors
    numerical_indices = tf.constant(numerical_indices, dtype=tf.int32)

    # Ensure output_sequence is a rank-1 tensor
    if len(output_sequence.shape) > 1:
        raise ValueError("Output sequence should be a rank-1 tensor.")

    # Convert output sequence to TensorFlow tensor and reshape it
    output_sequence_tensor = tf.convert_to_tensor(output_sequence, dtype=tf.int32)
    output_sequence_reshaped = tf.expand_dims(output_sequence_tensor, axis=0)

    # Solution 1: Failed
    # Create a tf.data.Dataset
    # Author: GPT3.5
    # dataset = tf.data.Dataset.from_tensor_slices((numerical_indices, output_sequence_reshaped))
    # error: Value tf.Tensor(0, shape=(), dtype=int32) has insufficient rank for batching.

    # Solution 2: Successful
    # Create a tf.data.Dataset
    # Author: Mike Aguilar
    # URL: https://copyprogramming.com/howto/valueerror-value-tensor-normalize-element-component-0-0-shape-dtype-int32-device-device-cpu-0-has-insufficient-rank-for-batching
    # Suppplemental: Bard AI provided implementation of solution 2 with creating a global `feature dictionary``.
    dataset = tf.data.Dataset.from_tensors(dict(FEATURES)).batch(batch_size)

    # Shuffle the dataset if specified
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(numerical_indices))

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    logging.info("[datasets] TensorFlow datasets created successfully.")

    return dataset