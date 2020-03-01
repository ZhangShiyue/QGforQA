import tensorflow as tf


def get_record_parser(config):
    para_limit = config.para_limit
    ques_limit = config.ques_limit

    def parse(example):
        features = tf.parse_single_example(example,
            features={
                "para_idxs": tf.FixedLenFeature([], tf.string),
                "para_idxs_unk": tf.FixedLenFeature([], tf.string),
                "ques_idxs": tf.FixedLenFeature([], tf.string),
                "labels": tf.FixedLenFeature([], tf.string),
                "pos_tags": tf.FixedLenFeature([], tf.string),
                "ner_tags": tf.FixedLenFeature([], tf.string),
                "id": tf.FixedLenFeature([], tf.int64)
            })

        para_idxs = tf.reshape(tf.decode_raw(
                features["para_idxs"], tf.int32), [para_limit])
        para_idxs_unk = tf.reshape(tf.decode_raw(
                features["para_idxs_unk"], tf.int32), [para_limit + 2])
        ques_idxs = tf.reshape(tf.decode_raw(
                features["ques_idxs"], tf.int32), [ques_limit + 2])
        labels = tf.reshape(tf.decode_raw(
                features["labels"], tf.int32), [para_limit])
        pos_tags = tf.reshape(tf.decode_raw(
                features["pos_tags"], tf.int32), [para_limit])
        ner_tags = tf.reshape(tf.decode_raw(
                features["ner_tags"], tf.int32), [para_limit])
        qa_id = features["id"]

        return para_idxs, para_idxs_unk, ques_idxs, labels, pos_tags, ner_tags, qa_id

    return parse