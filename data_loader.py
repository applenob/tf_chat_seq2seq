# coding=utf-8
# @author=cer
import os
import numpy as np
import math
import time
import sys
import jieba

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def get_batch(batch_size, data):
    """Given data, return a batch iterator.
    Data can be a array, or a tuple(list) of array."""
    s_index = 0
    e_index = batch_size
    if isinstance(data, np.ndarray):
        while e_index < len(data):
            batch = data[s_index: e_index]
            temp = e_index
            e_index = e_index + batch_size
            s_index = temp
            yield batch
    elif (isinstance(data, tuple) or isinstance(data, list)) \
            and isinstance(data[0], np.ndarray):
        while e_index < len(data[0]):
            batch = []
            for one in data:
                batch.append(one[s_index: e_index])
            temp = e_index
            e_index = e_index + batch_size
            s_index = temp
            yield batch
    else:
        print("check data type !!!")
        sys.exit(1)


def load_data(output_dir, config, data_type="train", use_word=False):
    s = time.time()
    if use_word:
        word_str = "word"
    else:
        word_str = "char"
    input_len = config["data"]["input_step"]
    encode_vec_name = f"{data_type}_encode.{word_str}.vec"
    decode_vec_name = f"{data_type}_decode.{word_str}.vec"
    encode_input_vec_path = os.path.join(output_dir, encode_vec_name)
    decode_input_vec_path = os.path.join(output_dir, decode_vec_name)
    encoder_inputs = []
    decoder_inputs = []
    print(f"Loading encode input vector from: {encode_input_vec_path}")
    with open(encode_input_vec_path, encoding="utf-8") as f:
        for line in f:
            encoder_inputs.append([int(one) for one in line.strip("\n").split()])
    print(f"Save decode input vector to: {decode_input_vec_path}")
    with open(decode_input_vec_path, encoding="utf-8") as f:
        for line in f:
            decoder_inputs.append([int(one) for one in line.strip("\n").split()])
    encoder_len = [len(one) for one in encoder_inputs]
    decoder_len = [len(one) + 1 for one in decoder_inputs]
    # print("max len of encoder:", max(encoder_len))
    # print("max len of decoder:", max(decoder_len))
    encoder_len = [one if one <=input_len else input_len for one in encoder_len]
    decoder_len = [one if one <=input_len else input_len for one in decoder_len]
    encoder_inputs_a = []
    decoder_inputs_a = []
    decoder_targets_a = []
    for one in encoder_inputs:
        if len(one) >= input_len:
            encoder_one = one[:input_len]
        else:
            encoder_one = one + [PAD_ID] * (input_len - len(one))
        encoder_inputs_a.append(encoder_one)
    for one in decoder_inputs:
        if len(one) >= input_len-1:
            dec_inp_one = [GO_ID] + one[:input_len-1]
            dec_tar_one = one[:input_len-1] + [EOS_ID]
        else:
            dec_inp_one = [GO_ID] + one + [PAD_ID] * (input_len - 1 - len(one))
            dec_tar_one = one + [EOS_ID] + [PAD_ID] * (input_len - 1 - len(one))
        decoder_inputs_a.append(dec_inp_one)
        decoder_targets_a.append(dec_tar_one)
    encoder_len = np.asarray(encoder_len)
    decoder_len = np.asarray(decoder_len)
    encoder_inputs_a = np.asarray(encoder_inputs_a)
    decoder_inputs_a = np.asarray(decoder_inputs_a)
    decoder_targets_a = np.asarray(decoder_targets_a)
    print(f"encoder_len: {np.shape(encoder_len)}")
    print(f"decoder_len: {np.shape(decoder_len)}")
    print(f"encoder_inputs_a: {np.shape(encoder_inputs_a)}")
    print(f"decoder_inputs_a: {np.shape(decoder_inputs_a)}")
    print(f"decoder_targets_a: {np.shape(decoder_targets_a)}")
    print(f"data loading time cost {time.time() - s:.2f}")
    return encoder_inputs_a, encoder_len, decoder_inputs_a, decoder_targets_a, decoder_len


def load_word_dict(word_dict_name="output/train_vocabulary.word"):
    tmp_vocab = []
    with open(word_dict_name, "r", encoding="utf-8") as input_f:
        tmp_vocab.extend(input_f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    word2id = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    return word2id


def one_question_to_input(question, word2id, use_word=True):
    if use_word:
        encoder_input = [word2id.get(word, UNK_ID) for word in jieba.cut(question)]
    else:
        encoder_input = [word2id.get(word, UNK_ID) for word in question]
    encoder_len = len(encoder_input)
    encoder_input = np.asarray([encoder_input])
    encoder_len = np.asarray([encoder_len])
    print(encoder_input)
    print(encoder_len)
    return encoder_input, encoder_len


def test_one_question_to_input():
    encoder_inputs_a, encoder_len, decoder_inputs_a, decoder_targets_a, decoder_len = load_data()
    c = 0
    for e1, el, d1, d2, dl in zip(encoder_inputs_a, encoder_len, decoder_inputs_a, decoder_targets_a, decoder_len):
        if el == 0 or dl == 0:
            c += 1
            print(el)
            print(e1)
            print(dl)
            print(d1)
            print(d2)
    print(c)


def test_iter():
    data = load_data()
    batch_iter = get_batch(5, data)
    encoder_inputs_a, encoder_len, decoder_inputs_a, decoder_targets_a, decoder_len = next(batch_iter)
    print(f"encoder_inputs_a: {encoder_inputs_a}")
    print(f"encoder_len: {encoder_len}")
    print(f"decoder_inputs_a: {decoder_inputs_a}")
    print(f"decoder_targets_a: {decoder_targets_a}")
    print(f"decoder_len: {decoder_len}")


if __name__ == '__main__':
    # test_one_question_to_input()
    test_iter()