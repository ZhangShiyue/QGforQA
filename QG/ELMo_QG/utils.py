import tensorflow as tf


def get_record_parser(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        char_limit = config.char_limit

        features = tf.parse_single_example(example,
                                           features={
                                               "para_idxs": tf.FixedLenFeature([], tf.string),
                                               "para_idxs_unk": tf.FixedLenFeature([], tf.string),
                                               "para_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs_unk": tf.FixedLenFeature([], tf.string),
                                               "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                                               "labels": tf.FixedLenFeature([], tf.string),
                                               "pos_tags": tf.FixedLenFeature([], tf.string),
                                               "ner_tags": tf.FixedLenFeature([], tf.string),
                                               "que_pos_tags": tf.FixedLenFeature([], tf.string),
                                               "que_ner_tags": tf.FixedLenFeature([], tf.string),
                                               "y1": tf.FixedLenFeature([], tf.string),
                                               "y2": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64)
                                           })

        para_idxs = tf.reshape(tf.decode_raw(
                features["para_idxs"], tf.int32), [para_limit])
        para_idxs_unk = tf.reshape(tf.decode_raw(
                features["para_idxs_unk"], tf.int32), [para_limit + 2])
        para_char_idxs = tf.reshape(tf.decode_raw(
                features["para_char_idxs"], tf.int32), [para_limit, char_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
                features["ques_idxs"], tf.int32), [ques_limit + 2])
        ques_idxs_unk = tf.reshape(tf.decode_raw(
                features["ques_idxs_unk"], tf.int32), [ques_limit + 2])
        ques_char_idxs = tf.reshape(tf.decode_raw(
                features["ques_char_idxs"], tf.int32), [ques_limit, char_limit])
        labels = tf.reshape(tf.decode_raw(
                features["labels"], tf.int32), [para_limit])
        pos_tags = tf.reshape(tf.decode_raw(
                features["pos_tags"], tf.int32), [para_limit])
        ner_tags = tf.reshape(tf.decode_raw(
                features["ner_tags"], tf.int32), [para_limit])
        que_labels = tf.reshape(tf.decode_raw(
                features["que_pos_tags"], tf.int32), [ques_limit])
        que_pos_tags = tf.reshape(tf.decode_raw(
                features["que_pos_tags"], tf.int32), [ques_limit])
        que_ner_tags = tf.reshape(tf.decode_raw(
                features["que_ner_tags"], tf.int32), [ques_limit])
        y1 = tf.reshape(tf.decode_raw(
                features["y1"], tf.float32), [para_limit])
        y2 = tf.reshape(tf.decode_raw(
                features["y2"], tf.float32), [para_limit])
        qa_id = features["id"]

        return para_idxs, para_idxs_unk, para_char_idxs, ques_idxs, ques_idxs_unk, ques_char_idxs,  \
               labels, pos_tags, ner_tags, que_labels, que_pos_tags, que_ner_tags, y1, y2, qa_id

    return parse
