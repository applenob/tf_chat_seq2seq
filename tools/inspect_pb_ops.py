# coding=utf-8
"""
Print all operations in a pb file.
"""

import sys
sys.path.append("..")
import argparse
from utils import *
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default="../pb/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph, sess = load_graph_session_from_pb(args.file_name)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions

    # # We access the input and output nodes
    # x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    # y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')
    #
    # # We launch a Session
    # with tf.Session(graph=graph) as sess:
    #     # Note: we don't nee to initialize/restore anything
    #     # There is no Variables in this graph, only hardcoded constants
    #     y_out = sess.run(y, feed_dict={
    #         x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]]  # < 45
    #     })
    #     # I taught a neural net to recognise when a sum of numbers is bigger than 45
    #     # it should return False in this case
    #     print(y_out)  # [[ False ]] Yay, it works!