# coding=utf-8
# @author: cer
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import sys
from tensorflow.contrib import layers
PAD_SYMBOL = 0
GO_SYMBOL = 1
END_SYMBOL = 2


class ChatModel:
    def __init__(self, vocab_size, config, embedding_matrix=None):
        self.hidden_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.beam_width = config["model"]["beam_width"]
        self.learning_rate = config["model"]["learning_rate"]
        self.max_decode_step = config["data"]["max_decode_step"]
        self.batch_size = config["data"]["batch_size"]
        self.use_pretrain_embed = config["model"]["use_pretrain_embed"]
        self.vocab_size = vocab_size

        # Model definition
        # [batch_size, steps]
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None],
                                             name='inputs')
        self.encoder_actual_length = tf.placeholder(tf.int32, [None],
                                                    name='encoder_actual_length')
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None],
                                             name='decoder_inputs')
        self.decoder_targets = tf.placeholder(tf.int32, [None, None],
                                              name='decode_targets')
        self.decoder_actual_length = tf.placeholder(tf.int32, [None],
                                                    name='decoder_actual_length')

        with tf.name_scope("embedding"):
            if self.use_pretrain_embed:
                init_embeddings = tf.constant(embedding_matrix, dtype=tf.float32)
                self.embeddings = tf.get_variable(name="embedding", initializer=init_embeddings,
                                                  dtype=tf.float32)
            else:
                self.embeddings = tf.get_variable(shape=[self.vocab_size, self.embedding_dim],
                                                  name='embedding_mat',
                                                  initializer=layers.xavier_initializer(),
                                                  dtype=tf.float32)

        self.encoder_inputs_tm = tf.transpose(self.encoder_inputs, perm=[1, 0])
        self.encoder_inputs_emb_tm = tf.nn.embedding_lookup(self.embeddings,
                                                            self.encoder_inputs_tm)
        self.decoder_inputs_tm = tf.transpose(self.decoder_inputs, perm=[1, 0])
        self.decoder_inputs_emb_tm = tf.nn.embedding_lookup(self.embeddings,
                                                            self.decoder_inputs_tm)
        self.decoder_targets_tm = tf.transpose(self.decoder_targets, perm=[1, 0])

        encoder_cell = LSTMCell(self.hidden_dim)
        # 下面变量的尺寸：T*B*D，B*D
        encoder_outputs, encoder_final_state = \
            tf.nn.dynamic_rnn(cell=encoder_cell,
                              inputs=self.encoder_inputs_emb_tm,
                              sequence_length=self.encoder_actual_length,
                              dtype=tf.float32, time_major=True)
        print("encoder_outputs: ", encoder_outputs)

        with tf.variable_scope("Decoder"):
            cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_dim)
            projection_layer = tf.layers.Dense(vocab_size, name="projection")
            # decoder for training
            memory = tf.transpose(encoder_outputs, [1, 0, 2])
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.hidden_dim,
                memory=memory,
                memory_sequence_length=self.encoder_actual_length)
            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.hidden_dim)

            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=self.decoder_inputs_emb_tm,
                sequence_length=self.decoder_actual_length,
                time_major=True)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=self.decoder_cell,
                helper=training_helper,
                initial_state=self.decoder_cell.zero_state(tf.shape(self.encoder_inputs)[0],
                                                           tf.float32).clone(cell_state=encoder_final_state),
                output_layer=projection_layer)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=training_decoder,
                impute_finished=True,
                maximum_iterations=self.max_decode_step,
                output_time_major=True)
            self.training_decoder_output = training_decoder_output

        with tf.variable_scope('Decoder', reuse=True):
            # decoder for predicting
            memory_tiled = tf.contrib.seq2seq.tile_batch(memory, self.beam_width)
            encoder_state_tiled = tf.contrib.seq2seq.tile_batch(encoder_final_state, self.beam_width)
            encoder_actual_length_tiled = tf.contrib.seq2seq.tile_batch(self.encoder_actual_length, self.beam_width)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.hidden_dim,
                memory=memory_tiled,
                memory_sequence_length=encoder_actual_length_tiled)
            self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.hidden_dim)

            predicting_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=self.decoder_cell,
                embedding=self.embeddings,
                start_tokens=tf.tile(tf.constant([GO_SYMBOL], dtype=tf.int32), [self.batch_size]),
                end_token=END_SYMBOL,
                initial_state=self.decoder_cell.zero_state(self.batch_size * self.beam_width, tf.float32).clone(
                    cell_state=encoder_state_tiled),
                beam_width=self.beam_width,
                output_layer=projection_layer,
                length_penalty_weight=0.0)
            predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=predicting_decoder,
                impute_finished=False,
                maximum_iterations=self.max_decode_step,
                output_time_major=True)
            predicting_ids = predicting_decoder_output.predicted_ids[:, :, 0]
            self.predicting_ids = tf.identity(predicting_ids, name="output_ids")

        # loss
        decoder_max_steps, _, _ = tf.unstack(tf.shape(training_decoder_output.rnn_output))
        self.decoder_targets_tm_cut = self.decoder_targets_tm[:decoder_max_steps]
        self.mask = tf.to_float(tf.not_equal(self.decoder_targets_tm_cut, PAD_SYMBOL))
        self.loss = tf.contrib.seq2seq.sequence_loss(
            training_decoder_output.rnn_output, self.decoder_targets_tm_cut, weights=self.mask)
        self.train_perplexity = tf.exp(self.loss)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 2.5)
        self.train_op = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)
        print("model built.")
        print("trainable_variables: ", tf.trainable_variables())
        print("attention variables: ", [var for var in tf.trainable_variables() if "attention" in var.op.name])

    def step(self, sess, mode, one_batch):
        """ perform each batch"""

        if mode == 'train':
            output_feeds = [self.train_op, self.loss, self.train_perplexity, self.predicting_ids]
            feed_dict = {self.encoder_inputs: one_batch[0],
                         self.encoder_actual_length: one_batch[1],
                         self.decoder_inputs: one_batch[2],
                         self.decoder_targets: one_batch[3],
                         self.decoder_actual_length: one_batch[4]}
        elif mode in ['test']:
            output_feeds = [self.predicting_ids]
            feed_dict = {self.encoder_inputs: one_batch[0],
                         self.encoder_actual_length: one_batch[1]}
        else:
            print('mode is not supported', file=sys.stderr)
            sys.exit(-1)

        results = sess.run(output_feeds, feed_dict=feed_dict)
        return results

    def save(self, session, checkpoint_path, checkpoint_name="best"):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            print("make dir: ", checkpoint_path)
        print('[*] saving checkpoints to {}...'.format(checkpoint_path))
        self.saver.save(session, os.path.join(checkpoint_path, checkpoint_name))

    def load(self, session, checkpoint_path, checkpoint_name="best"):
        print('[*] Loading checkpoints from {}...'.format(checkpoint_path))
        try:
            self.saver.restore(session, os.path.join(checkpoint_path, checkpoint_name))
        except Exception as e:
            print(e)
            print("check ckpt file path !!!")
            exit(1)
