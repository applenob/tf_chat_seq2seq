# coding=utf-8
# @author=cer
import tensorflow as tf
import os
import numpy as np
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from data_loader import *
from utils import *

print(tf.__version__)


def load_context(output_dir, use_word, pb_file="pb/frozen_model.pb"):
    if use_word:
        word_str = "word"
    else:
        word_str = "char"
    vocab_file = os.path.join(output_dir, f"train_vocab.{word_str}")
    with open(vocab_file) as f:
        vocab_size = len(f.readlines())
    print(f"vocab_file: {vocab_file}, vocab_size: {vocab_size}")

    word2id = load_word_dict(vocab_file)
    id2word = dict([(v, k) for (k, v) in word2id.items()])

    graph, sess = load_graph_session_from_pb(pb_file)
    encoder_inputs = graph.get_operation_by_name("import/inputs").outputs[0]
    encoder_actual_length = graph.get_operation_by_name("import/encoder_actual_length").outputs[0]
    predicting_ids = graph.get_operation_by_name("import/Decoder_1/output_ids").outputs[0]

    def predict_func(sess, one_batch):
        output_feeds = [predicting_ids]
        feed_dict = {encoder_inputs: one_batch[0],
                     encoder_actual_length: one_batch[1]
                     }
        return sess.run(output_feeds, feed_dict=feed_dict)
    return sess, predict_func, word2id, id2word


def sample_one_answer(sess, predict_func, word2id, id2word, question, use_word=True):
    test_batch = one_question_to_input(question, word2id, use_word=use_word)
    output = predict_func(sess, test_batch)
    # print(output, np.shape(output))
    output = np.transpose(np.squeeze(output))
    # print(output, np.shape(output))
    res = "".join([id2word.get(one, "__UNK__") for one in output])
    res = res.split("__EOS__")[0]
    return res


def sample_main(output_dir, use_word):
    sess, predict_func, word2id, id2word = load_context(output_dir, use_word)
    while True:
        try:
            question = input("请输入问题：")
        except UnicodeDecodeError as e:
            print(e)
            continue
        if question == "exit":
            print("exiting from the system.")
            exit(0)
        res = sample_one_answer(sess, predict_func, word2id, id2word, question, use_word)
        print("机器人：", res)


def test_time(output_dir, use_word):
    import time
    sess, predict_func, word2id, id2word = load_context(output_dir, use_word)
    question = "锄禾日当午"
    t = time.time()
    for i in range(10):
        t1 = time.time()
        res = sample_one_answer(sess, predict_func, word2id, id2word, question, use_word)
        print(f"Time cost: {time.time() - t1}")
        print(res)
    print(res)
    print(f"Total time cost: {time.time() - t}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sample",
                        help="sample/time, default sample")
    parser.add_argument("--output_dir", default="output",
                        help="output director.")
    parser.add_argument("--use_word", default=0, type=int,
                        help="if use word: 1, else: 0.")
    args = parser.parse_args()

    if args.mode == "sample":
        sample_main(args.output_dir, args.use_word)
    elif args.mode == "time":
        test_time(args.output_dir, args.use_word)
    else:
        print(f"Mode: {args.mode} not supported now!")
        exit(-1)
