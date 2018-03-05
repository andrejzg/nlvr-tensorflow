import tensorflow as tf
import numpy as np


def unidirectional_lstm(inputs, name, size, input_lengths=None):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        cell_fw = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=1.0)

        outputs, final_state = tf.nn.dynamic_rnn(
            cell=cell_fw,
            dtype=tf.float32,
            sequence_length=input_lengths,
            inputs=inputs
        )

    return outputs, final_state


def embed_sequence(inputs, name, vocab_size, trainable=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        embedding_matrix = tf.get_variable(
            name="embedding_matrix",
            initializer=tf.constant(np.eye(vocab_size),
                                    dtype=tf.float32,
                                    name="embedding_matrix_init"),
            trainable=trainable)

        embs = tf.nn.embedding_lookup(params=embedding_matrix,
                                      ids=inputs)
    return embs


def unpadded_lengths(tensor):
    """ input: batch size x input size x emb size """
    used = tf.sign(tensor)
    lengths = tf.reduce_sum(used, axis=-1)
    lengths = tf.cast(lengths, tf.int32)
    return lengths


def convert_to_text(tensor, id2word):
    """ input: batch size x input size """
    return tf.map_fn(lambda x: tf.map_fn(lambda i: id2word[i], x),
                     elems=tensor)
