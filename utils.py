# coding=utf-8
# @author: cer
import re
import yaml
import tensorflow as tf


def get_session():
    """load a new session"""
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    return tf.Session(config=sess_config)


def load_frozen_graph(frozen_graph_filename):
    """load a graph from protocol buffer file"""
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return graph


def load_graph_session_from_ckpt(ckpt_file):
    """load graph and session from checkpoint file"""
    graph = tf.Graph()
    with graph.as_default():
        sess = get_session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(ckpt_file))
            saver.restore(sess, ckpt_file)
    return graph, sess


def load_graph_session_from_pb(pb_file, print_op=False):
    """load graph and session from protocol buffer file"""
    graph = load_frozen_graph(pb_file)
    if print_op:
        for op in graph.get_operations():
            print(op.name)
    with graph.as_default():
        sess = get_session()
    return graph, sess


def contain_chinese(s):
    if re.findall('[\u4e00-\u9fa5]+', s):
        return True
    return False


def valid(a, max_len=0):
    if len(a) > 0 and contain_chinese(a):
        if max_len <= 0:
            return True
        elif len(a) <= max_len:
            return True
    return False


def load_yaml_config(yaml_file):
    with open(yaml_file) as f:
        config = yaml.load(f)
    return config
