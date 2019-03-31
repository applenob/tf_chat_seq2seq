# coding=utf-8
# @author=cer
import tensorflow as tf
import os
import numpy as np
from model_tf import ChatModel
from data_loader import *
from utils import load_yaml_config, get_session
from metrics import compute_bleu
from nltk.translate.bleu_score import corpus_bleu
config = load_yaml_config("config.yml")


def load_model(mode, output_dir, use_word):
    if use_word:
        word_str = "word"
    else:
        word_str = "char"
    vocab_file = os.path.join(output_dir, f"train_vocab.{word_str}")
    with open(vocab_file) as f:
        vocab_size = len(f.readlines())
    print(f"vocab_file: {vocab_file}, vocab_size: {vocab_size}")
    sess = get_session()
    model = ChatModel(vocab_size, config)
    epoch = config["model"]["saved_epoch"]
    if mode != "train" or config["model"]["resume_train"]:
        sess.run(tf.global_variables_initializer())
        model.load(sess, config["model"]["model_path"], f"epoch-{epoch}")
        print('Successfully load model!')
    else:
        sess.run(tf.global_variables_initializer())
        print('Init model using tf.global_variables_initializer()')
    return sess, model, vocab_file


def train(output_dir, use_word=False):
    sess, model, vocab_file = load_model("train", output_dir, use_word)

    best_train_loss = np.inf
    train_data = load_data(output_dir, config, use_word=use_word)
    batch_size = config["data"]["batch_size"]
    batch_num = int(len(train_data[0])/batch_size)
    print(f"Start training! total training data: {len(train_data[0])}, "
          f"total batch num: {batch_num} in one epoch.")
    max_epoch_num = config["model"]["max_epoch_num"]
    for epoch in range(max_epoch_num):
        train_loss = []
        ppl_s = []
        bleu_s = []
        s = time.time()
        s1 = time.time()
        train_batch_iter = get_batch(batch_size, train_data)
        for i, train_batch in enumerate(train_batch_iter):
            _, loss, perplexity, predicting_ids = model.step(sess, "train", train_batch)
            if i % 100 == 0:
                print(f"Epoch[{epoch}]: [{i} of {batch_num}], loss: {loss:.3f}, "
                      f"perplexity: {perplexity:.2f}, time cost: {time.time()- s1:.2f}s")
                sys.stdout.flush()
                s1 = time.time()
            train_loss.append(loss)
            ppl_s.append(perplexity)
            candidate_corpus = np.transpose(predicting_ids).tolist()
            reference_corpus = np.expand_dims(train_batch[2], axis=1).tolist()
            bleu_score = corpus_bleu(reference_corpus, candidate_corpus)
            # print(bleu_score)
            bleu_s.append(bleu_score)
        mean_loss = np.mean(train_loss)
        mean_ppl = np.mean(ppl_s)
        mean_bleu = np.mean(bleu_s)
        print(f"[Epoch {epoch}] Average loss: {mean_loss}, PPL: {mean_ppl}, "
              f"BLEU: {mean_bleu}, time cost: {time.time()- s:.2f}")
        if mean_loss < best_train_loss:
            print(f"Training loss decrease from {best_train_loss} to {mean_loss}")
            best_train_loss = mean_loss
            model_path = config["model"]["model_path"]
            print(f"Save new checkpoint  to: {model_path}")
            model.save(sess, model_path, f"epoch-{epoch}")
    sess.close()
    print("Model training done ! ")


def load_context(output_dir, use_word=True):
    sess, model, vocab_file = load_model("predict", output_dir, use_word)
    word2id = load_word_dict(vocab_file)
    id2word = dict([(v, k) for (k, v) in word2id.items()])
    return sess, model, word2id, id2word


def sample_one_answer(sess, model, word2id, id2word, question, use_word=True):
    test_batch = one_question_to_input(question, word2id, use_word=use_word)
    output = model.step(sess, "test", test_batch)
    # print(output, np.shape(output))
    output = np.transpose(np.squeeze(output))
    # print(output, np.shape(output))
    res = "".join([id2word.get(one, "__UNK__") for one in output])
    res = res.split("__EOS__")[0]
    return res


def sample_main(output_dir, use_word=True):
    sess, model, word2id, id2word = load_context(output_dir, use_word=use_word)
    while True:
        try:
            question = input("请输入问题：")
        except UnicodeDecodeError as e:
            print(e)
            continue
        if question == "exit":
            print("exiting from the system.")
            exit(0)
        res = sample_one_answer(sess, model, word2id, id2word, question, use_word)
        print("机器人：", res)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train",
                        help="train or predict, default train")
    parser.add_argument("--output_dir", default="output",
                        help="output director.")
    parser.add_argument("--use_word", default=0, type=int,
                        help="if use word: 1, else: 0.")
    args = parser.parse_args()
    use_word = bool(args.use_word)
    if args.mode == "train":
        train(output_dir=args.output_dir, use_word=use_word)
    else:
        sample_main(output_dir=args.output_dir, use_word=use_word)

