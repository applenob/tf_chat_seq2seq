# coding=utf-8
"""
Print all variables in a checkpoint file.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from utils import *

print(tf.__version__)

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="ckpt", type=str,
                        help="The folder of ckpt.")
    args = parser.parse_args()

    checkpoint = tf.train.get_checkpoint_state(args.ckpt)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We use our "load_graph" function
    graph, sess = load_graph_session_from_ckpt(input_checkpoint)

    # We can verify that we can access the list of operations in the graph
    with graph.as_default():
        for var in tf.global_variables():
            print(var.name, var.shape)
