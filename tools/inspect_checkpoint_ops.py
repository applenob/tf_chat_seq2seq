# coding=utf-8
"""
Print all operations in a checkpoint file.
"""

import sys
sys.path.append("..")
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

    # We use our "load_graph" function
    graph, sess = load_graph_session_from_ckpt(args.ckpt)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
