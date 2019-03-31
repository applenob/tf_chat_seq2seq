# coding=utf-8
"""
Freeze a model from ckpt to pb format.
"""
import sys
sys.path.append("..")
import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.python.tools.optimize_for_inference_lib import node_name_from_input


def freeze_graph(ckpt_folder, pb_folder, pb_name, output_node_names):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(ckpt_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    if not os.path.exists(pb_folder):
        os.makedirs(pb_folder)
    output_graph = os.path.join(pb_folder, pb_name)

    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    print(f"Loading ckpt file: {input_checkpoint}.meta")
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    # input_graph_def = strip_new(input_graph_def)
    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Checkpoint model folder")
    parser.add_argument("--pb", type=str, help="Pb model folder")
    parser.add_argument("--pb_name", type=str, default="frozen_model.pb",
                        help="Pb model file name.")
    parser.add_argument("--output_nodes", type=str, default="Decoder_1/output_ids",
                        help="Output nodes of the graph, split by comma.")
    args = parser.parse_args()

    freeze_graph(args.ckpt, args.pb, args.pb_name, args.output_nodes)
