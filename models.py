import ops

import tensorflow as tf


def simple_conv_lstm(image, question, config):
    # Embed the image
    image_emb_1 = tf.layers.conv2d(inputs=image,
                                   filters=10,
                                   kernel_size=[10, 10],
                                   strides=(1, 1),
                                   activation=tf.nn.relu)

    image_emb_2 = tf.layers.conv2d(inputs=image_emb_1,
                                   filters=10,
                                   kernel_size=[10, 10],
                                   strides=(1, 1),
                                   activation=tf.nn.relu)

    image_emb_3 = tf.layers.conv2d(inputs=image_emb_2,
                                   filters=10,
                                   kernel_size=[10, 10],
                                   strides=(1, 1),
                                   activation=tf.nn.relu)

    pooled = tf.layers.max_pooling2d(inputs=image_emb_3,
                                     pool_size=[5, 5],
                                     strides=[1, 1])

    image_emb = tf.contrib.layers.flatten(pooled)

    # Embed the question
    question_emb = ops.embed_sequence(inputs=question, vocab_size=config['vocab_size'], name='embedder')

    question_lstm_emb, _ = ops.unidirectional_lstm(inputs=question_emb,
                                                   input_lengths=ops.unpadded_lengths(question),
                                                   name='question_text_LSTM',
                                                   size=50)

    question_pooled = tf.reduce_mean(input_tensor=question_lstm_emb,
                                     axis=1)

    img_question = tf.concat([image_emb, question_pooled], axis=-1)

    res = tf.layers.dense(inputs=img_question,
                          units=2,
                          use_bias=True,
                          name='final_dense_layer')

    return res
