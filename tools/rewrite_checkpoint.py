# coding=utf-8
"""修改checkpoint的图结构（非var部分）"""
import sys
sys.path.append("..")
from tf_model import ChatModel
import os
import argparse
from utils import *
vocab_size = 3794
word_str = "char"
model_path = "saved_model"
config = load_yaml_config("../config.yml")


def load_new_model(sess):
    model = ChatModel(vocab_size, config)
    model.load(sess, model_path, word_str)
    return model


if __name__ == '__main__':
    sess = get_session()
    new_model = load_new_model(sess)
    new_model.save(sess, model_path, word_str + ".new")
