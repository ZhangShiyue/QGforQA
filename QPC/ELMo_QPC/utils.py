import tensorflow as tf


def get_record_parser_qqp(config, is_test=False):
    def parse(example):
        ques_limit = config.test_ques_limit if is_test else config.ques_limit

        features = tf.parse_single_example(example,
                                           features={
                                               "ques1_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques2_idxs": tf.FixedLenFeature([], tf.string),
                                               "label": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })

        ques1_idxs = tf.reshape(tf.decode_raw(
                features["ques1_idxs"], tf.int32), [ques_limit + 2])
        ques2_idxs = tf.reshape(tf.decode_raw(
                features["ques2_idxs"], tf.int32), [ques_limit + 2])
        label = tf.reshape(tf.decode_raw(
                features["label"], tf.float32), [2])
        qa_id = features["id"]

        return ques1_idxs, ques2_idxs, label, qa_id

    return parse
