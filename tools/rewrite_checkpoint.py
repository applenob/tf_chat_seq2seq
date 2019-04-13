# coding=utf-8
"""修改checkpoint的图结构（非var部分）"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_tf import ChatModel
import os
import argparse
from utils import *
vocab_size = 3794
word_str = "char"
config = load_yaml_config("config.yml")
model_path = config["model"]["saved_model"]


def load_new_model(sess):
    model = ChatModel(vocab_size, config)
    model.load(sess, model_path, word_str)
    return model


if __name__ == '__main__':
    sess = get_session()
    new_model = load_new_model(sess)
    new_model.save(sess, model_path, word_str + ".new")
