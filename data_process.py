# coding=utf-8
# @author=cer

import os
import random
import argparse
from collections import defaultdict
import jieba
from utils import *
config = load_yaml_config("config.yml")


# special tokens
PAD = "__PAD__"  # padding token
GO = "__GO__"  # decoder start token
EOS = "__EOS__"  # decoder end token
UNK = "__UNK__"  # unknown token
START_VOCABULARY = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
user_dict_file = "input/user_dict.txt"
if os.path.exists(user_dict_file):
    jieba.load_userdict(user_dict_file)


def split_dataset(conv_path):
    """
    Split conversation file into text questions and answers.
    conversation data format:
    ```
    E
    M 不是
    M 那是什么？
    E
    M 怎么了
    M 我很难过，安慰我~
    E
    ```
    E: split conversations
    M: contents of conversations
    """
    print("Start splitting ...")
    convs = []  # conversations
    with open(conv_path, encoding="utf8", errors='ignore') as f:
        one_conv = []  # one conversation
        for line in f:
            line = line.strip()
            if line == '':
                continue
            if line[0] == 'E':
                if one_conv:
                    convs.append(one_conv)
                one_conv = []
            elif line[0] == 'M' and len(line.split(' ')) >= 2:
                one_conv.append(line.split(' ')[1].replace("/", ""))

    # split conversation into questions and answers
    questions = []
    answers = []
    for conv in convs:
        if len(conv) == 1:
            continue
        if len(conv) % 2 != 0:
            conv = conv[:-1]
        for i in range(int(len(conv) / 2)):
            # ask and response must both be valid.
            if valid(conv[2 * i]) and valid(conv[2 * i + 1]):
                questions.append(conv[2 * i])
                answers.append(conv[2 * i + 1])
    print("Splitting done.")
    return questions, answers


def save_seq2seq_files(questions, answers, output_dir, test_set_size=8000):
    """
    Save text questions and answers.
    output
    *.enc: conversation inputs
    *.dec: conversation outputs
    """

    print("Saving seq2seq file ...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Make dir: {output_dir}")

    train_enc_file = os.path.join(output_dir, "train.enc")
    train_dec_file = os.path.join(output_dir, "train.dec")
    test_enc_file = os.path.join(output_dir, "test.enc")
    test_dec_file = os.path.join(output_dir, "test.dec")
    train_encode_decode_file = os.path.join(output_dir, "train.enc+dec")

    train_enc = open(train_enc_file, 'w')
    train_dec = open(train_dec_file, 'w')
    test_enc = open(test_enc_file, 'w')
    test_dec = open(test_dec_file, 'w')
    train_enc_dec = open(train_encode_decode_file, "w")

    print(f"Saving train encode data to: {train_enc_file}")
    print(f"Saving train decode data to: {train_dec_file}")
    print(f"Saving test encode data to: {test_enc_file}")
    print(f"Saving test decode data to: {test_dec_file}")
    print(f"Saving train encode and decode data to: {train_encode_decode_file}")

    test_index = random.sample([i for i in range(len(questions))], test_set_size)
    test_index = set(test_index)
    for i in range(len(questions)):
        if i in test_index:
            test_enc.write(questions[i] + '\n')
            test_dec.write(answers[i] + '\n')
        else:
            train_enc.write(questions[i] + '\n')
            train_dec.write(answers[i] + '\n')
            train_enc_dec.write(questions[i] + '\n')
            train_enc_dec.write(answers[i] + '\n')
        if i % 10000 == 0:
            print(f"Saving process: {i} of {len(range(len(questions)))}")

    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()
    print("Seq2seq file saved.")


def tokenize(output_dir, use_word=False):
    """
    Tokenize text file and save to vector file.
    """
    train_encode_file = os.path.join(output_dir, "train.enc")
    train_decode_file = os.path.join(output_dir, "train.dec")
    test_encode_file = os.path.join(output_dir, "test.enc")
    test_decode_file = os.path.join(output_dir, "test.dec")
    train_encode_decode_file = os.path.join(output_dir, "train.enc+dec")

    print('Start building vocabs ...')
    if use_word:
        use_word_str = "word"
    else:
        use_word_str = "char"
    train_vocabulary_file = os.path.join(output_dir, 'train_vocab.' + use_word_str)
    gen_vocabulary_file(train_encode_decode_file, train_vocabulary_file,
                        use_word=use_word)

    print("Start tokenizing...")

    train_enc_vec_file = f'output/train_encode.{use_word_str}.vec'
    train_dec_vec_file= f'output/train_decode.{use_word_str}.vec'
    test_enc_vec_file = f'output/test_encode.{use_word_str}.vec'
    test_dec_vec_file = f'output/test_decode.{use_word_str}.vec'

    tokenize_a_file(train_encode_file, train_vocabulary_file,
                    train_enc_vec_file,
                    use_word=use_word)
    tokenize_a_file(train_decode_file, train_vocabulary_file,
                    train_dec_vec_file,
                    use_word=use_word)
    tokenize_a_file(test_encode_file, train_vocabulary_file,
                    test_enc_vec_file,
                    use_word=use_word)
    tokenize_a_file(test_decode_file, train_vocabulary_file,
                    test_dec_vec_file,
                    use_word=use_word)
    print("Tokenizing done.")


def gen_vocabulary_file(input_file, output_file, use_word=False):
    word_freq = defaultdict(int)
    with open(input_file) as f:
        for line in f:
            if use_word:
                import jieba
                jieba.load_userdict(user_dict_file)
                tokens = list(jieba.cut(line.strip().strip("\n").replace("\u3000", "")))
            else:
                tokens = [char for char in line.strip().strip("\n").replace("\u3000", "")]
            for word in tokens:
                word_freq[word] += 1
    min_freq = config["data"]["vocab_min_freq"]
    del_words = []
    for word in word_freq:
        if word_freq[word] < min_freq:
            del_words.append(word)
    for word in del_words:
        del word_freq[word]
    vocabulary_list = START_VOCABULARY + sorted(word_freq.keys(),
                                                key=lambda w: word_freq[w],
                                                reverse=True)
    # keep the most frequent words
    vocab_max_size = config["data"]["vocab_max_size"]
    if len(vocabulary_list) > vocab_max_size:
        vocabulary_list = vocabulary_list[:vocab_max_size]
    print(f"Vocab size of {input_file}: {len(vocabulary_list)}")
    print(f"Save vocab to: {output_file}")
    with open(output_file, "w") as ff:
        for word in vocabulary_list:
            ff.write(word + "\n")


def tokenize_a_file(input_file, vocabulary_file, output_file, use_word=False):
    print(f"Tokenize vector data to: {output_file}")
    tmp_vocab = []
    with open(vocabulary_file, "r") as input_f:
        tmp_vocab.extend(input_f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    word2id = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    with open(output_file, 'w') as output_f, open(input_file, 'r') as input_f:
        unk_tokens = []
        for line in input_f:
            line_vec = []
            if use_word:
                tokens = list(jieba.cut(line.replace("\u3000", "").strip().strip("\n")))
            else:
                tokens = [char.strip() for char in line.replace("\u3000", "").strip().strip("\n")]
            for token in tokens:
                if token not in word2id:
                    unk_tokens.append(token)
                line_vec.append(word2id.get(token, UNK_ID))
            output_f.write(" ".join([str(num) for num in line_vec]) + "\n")
    print(f"token: {' '.join(unk_tokens)} not in vocab.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conv_path", default="input/xiaohuangji50w_nofenci.conv",
                        help="raw conversation data path.")
    parser.add_argument("--output_dir", default="output",
                        help="output director.")
    parser.add_argument("--use_word", default="0", type=int,
                        help="if use word: 1; else: 0.")
    args = parser.parse_args()
    if args.conv_path == "":
        print("Please check your args, 'conv_path' is needed!")
        exit(-1)
    use_word = bool(args.use_word)
    questions, answers = split_dataset(args.conv_path)
    save_seq2seq_files(questions, answers, args.output_dir)
    tokenize(args.output_dir, use_word=use_word)
