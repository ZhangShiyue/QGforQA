import sys
sys.path.append('../..')
import numpy as np
import ujson as json
import tensorflow as tf
from tqdm import tqdm
from LIB.utils import save


def process_file(qqppath, lower_word=False):
    examples = []
    eval_examples = {}
    total, total_p = 0, 0
    max_q = 0
    with open("{}.tok".format(qqppath), 'r') as fq:
        while True:
            total += 1
            line = fq.readline()
            if not line:
                break
            que1, que2, label = line.strip().split('\t')
            if not que1 or not que2 or not label:
                continue
            if lower_word:
                que1 = que1.lower()
                que2 = que2.lower()
            que1_tokens = ["<S>"] + [w for w in que1.strip().split(' ')] + ["</S>"]
            que2_tokens = ["<S>"] + [w for w in que2.strip().split(' ')] + ["</S>"]
            max_q = max(max_q, len(que1_tokens))
            max_q = max(max_q, len(que2_tokens))
            label = int(label)
            total_p += label
            example = {"question1": que1_tokens, "question2": que2_tokens, "label": label, "id": total}
            eval_examples[str(total)] = {"question1": que1_tokens, "question2": que2_tokens, "label": label}
            examples.append(example)
        np.random.shuffle(examples)
        print("{} examples in total, {} positive examples".format(len(examples), total_p))
        print("max_q: {}".format(max_q))
    return examples, eval_examples


def build_features(config, examples, data_type, out_file, word2idx_dict, is_test=False):
    ques_limit = config.test_ques_limit if is_test else config.ques_limit

    def filter_func(example):
        return len(example["question1"]) > ques_limit + 2 or \
               len(example["question2"]) > ques_limit + 2

    def _get_word(word):
        if word in word2idx_dict:
            return word2idx_dict[word]
        return 1

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example):
            continue

        total += 1

        ques1_idxs = np.zeros([ques_limit + 2], dtype=np.int32)
        ques2_idxs = np.zeros([ques_limit + 2], dtype=np.int32)
        label = np.zeros([2], dtype=np.float32)

        label[int(example["label"])] = 1.0

        for i, token in enumerate(example["question1"]):
            wid = _get_word(token)
            ques1_idxs[i] = wid

        for i, token in enumerate(example["question2"]):
            wid = _get_word(token)
            ques2_idxs[i] = wid

        record = tf.train.Example(features=tf.train.Features(feature={
            "ques1_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques1_idxs.tostring()])),
            "ques2_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques2_idxs.tostring()])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
        }))
        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def prepare(config):
    # process files
    train_examples, train_eval = process_file(config.train_file, lower_word=config.lower_word)
    dev_examples, dev_eval = process_file(config.dev_file, lower_word=config.lower_word)

    with open(config.word_dictionary, "r") as fh:
        word2idx_dict = json.load(fh)
    train_meta = build_features(config, train_examples, "train", config.train_record_file,
                                word2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev", config.dev_record_file,
                              word2idx_dict)

    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.train_meta, train_meta, message="train meta")
    save(config.dev_meta, dev_meta, message="dev meta")