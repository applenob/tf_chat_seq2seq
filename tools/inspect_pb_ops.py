# coding=utf-8
"""
Print all operations in a pb file.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from utils import *
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default="pb/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph, sess = load_graph_session_from_pb(args.file_name)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)

