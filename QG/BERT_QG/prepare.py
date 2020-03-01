import numpy as np
import ujson as json
import tensorflow as tf
from tqdm import tqdm
import sys
sys.path.append('../..')

from LIB.utils import save


def process_file(parapath, questionpath, para_limit):
    examples = []
    eval_examples = {}
    total = 0
    count = 0
    max_p, max_q = 0, 0
    with open("{}.label".format(parapath), 'r') as fpl, open("{}.tok".format(parapath), 'r') as fp, \
            open("{}.pos".format(parapath), 'r') as fpp, open("{}.ner".format(parapath), 'r') as fpn, \
            open("{}.tok".format(questionpath), 'r') as fq, open("{}_eval.tok".format(questionpath), 'r') as fqe,\
            open("{}.pos".format(questionpath), 'r') as fqp, open("{}.ner".format(questionpath), 'r') as fqn:
        while True:
            total += 1
            label, para, pos, ner = fpl.readline(), fp.readline(), fpp.readline(), fpn.readline()
            question, question_eval, que_pos, que_ner = fq.readline(), fqe.readline(), fqp.readline(), fqn.readline()
            if not para:
                break
            para_tokens = para.strip().split(' ')
            labels = label.strip().split(' ')
            pos_tags = pos.strip().split(' ')
            ner_tags = ner.strip().split(' ')
            start = labels.index('B')
            if 'O' in labels[start:]:
                end = start + labels[start:].index('O')
            else:
                end = len(labels)
            answer_tokens = para_tokens[start: end]
            if len(para_tokens) > para_limit:
                ans_len = end - start
                half = (para_limit - ans_len) // 2
                low, high = max(0, start - half), min(len(para_tokens), end + half)
                para_tokens = para_tokens[low: high]
                labels = labels[low: high]
                pos_tags = pos_tags[low: high]
                ner_tags = ner_tags[low: high]
                count += 1
            start = labels.index('B')
            if 'O' in labels[start:]:
                end = start + labels[start:].index('O')
            else:
                end = len(labels)
            end -= 1
            assert len(para_tokens) == len(labels) == len(pos_tags) == len(ner_tags) <= para_limit
            que_tokens = question.strip().split(' ')
            que_tokens_eval = question_eval.strip().split(' ')

            max_p = max(max_p, len(para_tokens))
            max_q = max(max_q, len(que_tokens))

            example = {"paragraph": para_tokens, "labels": labels, "pos_tags": pos_tags, "ner_tags": ner_tags,
                       "question": que_tokens, "answer": answer_tokens, "start": start, "end": end, "id": total}
            eval_examples[str(total)] = {"question": que_tokens, "paragraph": para_tokens, "answer": answer_tokens,
                                         "question_eval": que_tokens_eval, "start": start, "end": end}
            examples.append(example)
        np.random.shuffle(examples)
        print("{} examples in total".format(len(examples)))
        print("max_p, max_q: {}, {}".format(max_p, max_q))
        print("{} truncated examples".format(count))
    return examples, eval_examples


def build_features(config, examples, data_type, out_file, word2idx_dict, pos2idx_dict,
                   ner2idx_dict, label2idx_dict, para_limit, ques_limit, max_length):

    def filter_func(example):
        return len(example["paragraph"]) > para_limit or \
               len(example["question"]) > ques_limit

    def _get_word(word):
        if word in word2idx_dict:
            return word2idx_dict[word]
        return word2idx_dict["[UNK]"]

    def _get_pos(pos):
        if pos in pos2idx_dict:
            return pos2idx_dict[pos]
        return 1

    def _get_ner(ner):
        if ner in ner2idx_dict:
            return ner2idx_dict[ner]
        return 1

    def _get_label(label):
        if label in label2idx_dict:
            return label2idx_dict[label]
        return 1

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total, total_ = 0, 0
    meta = {}
    for example in tqdm(examples):
        total += 1

        if filter_func(example):
            continue

        total_ += 1

        # qg
        para_idxs = np.zeros([para_limit], dtype=np.int32)
        para_idxs_unk = np.zeros([para_limit + 2], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit + 2], dtype=np.int32)
        labels = np.zeros([para_limit], dtype=np.int32)
        pos_tags = np.zeros([para_limit], dtype=np.int32)
        ner_tags = np.zeros([para_limit], dtype=np.int32)

        for i, token in enumerate(example["paragraph"]):
            wid = _get_word(token)
            para_idxs[i] = len(word2idx_dict) + i if wid == 100 else wid

        para_tokens = ["[CLS]"] + example["paragraph"] + ["[SEP]"]
        for i, token in enumerate(para_tokens):
            wid = _get_word(token)
            para_idxs_unk[i] = wid

        que_tokens = ["[CLS]"] + example["question"] + ["[SEP]"]
        for i, token in enumerate(que_tokens):
            wid = _get_word(token)
            ques_idxs[i] = len(word2idx_dict) + example["paragraph"].index(token) \
                if wid == 100 and token in example["paragraph"] else wid

        for i, token in enumerate(example["labels"]):
            wid = _get_label(token)
            labels[i] = wid

        for i, token in enumerate(example["pos_tags"]):
            wid = _get_pos(token)
            pos_tags[i] = wid

        for i, token in enumerate(example["ner_tags"]):
            wid = _get_ner(token)
            ner_tags[i] = wid

        record = tf.train.Example(features=tf.train.Features(feature={
            "para_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[para_idxs.tostring()])),
            "para_idxs_unk": tf.train.Feature(bytes_list=tf.train.BytesList(value=[para_idxs_unk.tostring()])),
            "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
            "labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tostring()])),
            "pos_tags": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pos_tags.tostring()])),
            "ner_tags": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ner_tags.tostring()])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
        }))
        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total_, total))
    meta["total"] = total
    writer.close()
    return meta


def prepare(config):
    # process files
    train_examples, train_eval = process_file(config.train_para_file, config.train_question_file, para_limit=config.para_limit)
    dev_examples, dev_eval = process_file(config.dev_para_file, config.dev_question_file, para_limit=config.para_limit)
    test_examples, test_eval = process_file(config.test_para_file, config.test_question_file, para_limit=config.para_limit)

    with open(config.word_dictionary, "r") as fh:
        word2idx_dict = json.load(fh)
        print("num of words {}".format(len(word2idx_dict)))
    with open(config.label_dictionary, "r") as fh:
        label2idx_dict = json.load(fh)
        print("num of labels {}".format(len(label2idx_dict)))
    with open(config.pos_dictionary, "r") as fh:
        pos2idx_dict = json.load(fh)
        print("num of pos tags {}".format(len(pos2idx_dict)))
    with open(config.ner_dictionary, "r") as fh:
        ner2idx_dict = json.load(fh)
        print("num of ner tags {}".format(len(ner2idx_dict)))

    train_meta = build_features(config, train_examples, "train", config.train_record_file, word2idx_dict,
                                pos2idx_dict, ner2idx_dict, label2idx_dict, config.para_limit,
                                   config.ques_limit, config.max_input_length)
    dev_meta = build_features(config, dev_examples, "dev", config.dev_record_file, word2idx_dict, pos2idx_dict,
                              ner2idx_dict, label2idx_dict, config.para_limit,
                              config.ques_limit, config.max_input_length)
    test_meta = build_features(config, test_examples, "test", config.test_record_file, word2idx_dict, pos2idx_dict,
                               ner2idx_dict, label2idx_dict, config.para_limit,
                               config.ques_limit, config.max_input_length)

    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.train_meta, train_meta, message="train meta")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")