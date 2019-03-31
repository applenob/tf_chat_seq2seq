# coding=utf-8
# @author: cer
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import sys
from tensorflow.contrib import layers
from utils import load_yaml_config

PAD_SYMBOL = 0
GO_SYMBOL = 1
END_SYMBOL = 2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, inputs):
        print(f"{self.name} inputs: {inputs}")
        x, hidden = inputs
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def build(self, input_sahpe):
        super(Encoder, self).build(input_sahpe)
        self.built = True

    def compute_output_shape(self, input_shape):
        print(f"{self.name} input_shape: {input_shape}")
        return [tf.TensorShape([self.batch_sz, self.enc_units]),
                tf.TensorShape([self.batch_sz, self.enc_units])]

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def build(self, input_sahpe):
        super(BahdanauAttention, self).build(input_sahpe)
        self.built = True

    def compute_output_shape(self, input_shape):
        print(f"{self.name} input_shape: {input_shape}")
        batch_sz = input_shape[0][0]
        hidden_size = input_shape[0][1]
        return [tf.TensorShape([batch_sz, hidden_size]),
                tf.TensorShape([batch_sz, hidden_size])]

    def call(self, inputs):
        print(f"{self.name} inputs: {inputs}")
        query, values = inputs
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.vocab_size = vocab_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def build(self, input_sahpe):
        super(Decoder, self).build(input_sahpe)
        self.built = True

    def compute_output_shape(self, input_shape):
        print(f"{self.name} input_shape: {input_shape}")
        batch_sz = input_shape[0][0]
        hidden_size = input_shape[1][1]
        return [tf.TensorShape([batch_sz, self.vocab_size]),
                tf.TensorShape([batch_sz, hidden_size]),
                tf.TensorShape([batch_sz, hidden_size])]

    def call(self, inputs):
        print(f"{self.name} inputs: {inputs}")
        x, hidden, enc_output = inputs
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention([hidden, enc_output])

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


class ChatModel(tf.keras.Model):
    def __init__(self, vocab_size, config, embedding_matrix=None, **kwargs):
        super(ChatModel, self).__init__(kwargs)
        self.hidden_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.beam_width = config["model"]["beam_width"]
        self.learning_rate = config["model"]["learning_rate"]
        self.max_decode_step = config["data"]["max_decode_step"]
        self.batch_size = config["data"]["batch_size"]
        self.use_pretrain_embed = config["model"]["use_pretrain_embed"]
        self.vocab_size = vocab_size

        self.encoder = Encoder(self.vocab_size, self.embedding_dim,
                               self.hidden_dim, self.batch_size)
        self.attention_layer = BahdanauAttention(10)
        self.decoder = Decoder(self.vocab_size, self.embedding_dim,
                               self.hidden_dim, self.batch_size)

    def call(self, x, mask=None):
        # sample input
        print(f"{self.name} inputs: {x}")
        sample_hidden = self.encoder.initialize_hidden_state()
        sample_output, sample_hidden = self.encoder([x, sample_hidden])
        print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
        print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

        attention_result, attention_weights = self.attention_layer([sample_hidden, sample_output])

        print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
        print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

        sample_decoder_output, _, _ = self.decoder([tf.random.uniform((64, 1)),
                                                   sample_hidden, sample_output])
        print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
        return sample_decoder_output


if __name__ == '__main__':
    vocab_size = 3794
    config = load_yaml_config("config.yml")
    model = ChatModel(vocab_size, config)
    mock_input = np.random.randint(0, 3794, [64, 30])
    print("mock_input: ", mock_input)
    model(tf.convert_to_tensor(mock_input))
    print(model.summary())
