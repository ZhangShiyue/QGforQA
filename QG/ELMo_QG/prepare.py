"""
Data preparation
"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import ujson as json
import sys
sys.path.append('../..')

from LIB.utils import save


def process_file(parapath, questionpath, answerpath, lower_word=True):
    examples = []
    eval_examples = {}
    total = 0
    max_p, max_q, max_a = 0, 0, 0
    with open("{}.label".format(parapath), 'r') as fl, open("{}.tok".format(parapath), 'r') as fp, \
            open("{}.tok".format(questionpath), 'r') as fq, open("{}.tok".format(answerpath), 'r') as fa, \
            open("{}.pos".format(parapath), 'r') as fpp, open("{}.ner".format(parapath), 'r') as fpn, \
            open("{}.pos".format(questionpath), 'r') as fqp, open("{}.ner".format(questionpath), 'r') as fqn:
        while True:
            total += 1
            labels, para, question, answer = fl.readline(), fp.readline(), fq.readline(), fa.readline()
            if lower_word:
                para, question, answer = para.lower(), question.lower(), answer.lower()
            pos, ner = fpp.readline(), fpn.readline()
            que_pos, que_ner = fqp.readline(), fqn.readline()
            if not para:
                break
            para_tokens = [w for w in para.strip().split(' ')]
            para_tokens_unk = ["<S>"] + [w for w in para.strip().split(' ')] + ["</S>"]
            ques_tokens = ["<S>"] + [w for w in question.strip().split(' ')] + ["</S>"]
            que_labels = ['O' for _ in range(len(ques_tokens) - 2)]
            pos_tags = pos.strip().split(' ')
            ner_tags = ner.strip().split(' ')
            que_pos_tags = que_pos.strip().split(' ')
            que_ner_tags = que_ner.strip().split(' ')
            max_p = max(max_p, len(para_tokens))
            max_q = max(max_q, len(ques_tokens))

            answers = answer.strip().split('\t')
            all_ans_tokens = []
            for answer in answers:
                ans_tokens = [w for w in answer.split(' ')]
                max_a = max(max_a, len(ans_tokens))
                all_ans_tokens.append(ans_tokens)

            labels = labels.strip().split(' ')
            answer_start = labels.index("B")
            answer_end = answer_start + len(all_ans_tokens[0]) - 1

            assert len(para_tokens) == len(labels) == len(pos_tags) == len(ner_tags)

            example = {"paragraph": para_tokens, "paragraph_unk": para_tokens_unk, "pos_tags": pos_tags,
                       "ner_tags": ner_tags, "labels": labels, "question": ques_tokens, "que_labels": que_labels,
                       "que_pos_tags": que_pos_tags, "que_ner_tags": que_ner_tags, "answer": all_ans_tokens,
                       "answer_start": answer_start, "answer_end": answer_end, "id": total}
            eval_examples[str(total)] = {"question": ques_tokens, "paragraph": para_tokens, "answer": all_ans_tokens,
                                         "answer_start": answer_start, "answer_end": answer_end}
            examples.append(example)
        np.random.shuffle(examples)
        print("{} examples in total".format(len(examples)))
        print("max_p, max_q, max_a: {}, {}, {}".format(max_p, max_q, max_a))
    return examples, eval_examples


def build_features(config, examples, data_type, out_file, word2idx_dict, pos2idx_dict,
                   ner2idx_dict, label2idx_dict, char2idx_dict, is_test=False):
    para_limit = config.test_para_limit if is_test else config.para_limit
    ques_limit = config.test_ques_limit if is_test else config.ques_limit
    ans_limit = config.test_ans_limit if is_test else config.ans_limit
    char_limit = config.char_limit

    def filter_func(example):
        return len(example["paragraph"]) > para_limit + 2 or \
               len(example["question"]) > ques_limit + 2 or \
               len(example["answer"][0]) > ans_limit

    def _get_id(key, key2idx_dict):
        if key in key2idx_dict:
            return key2idx_dict[key]
        return 1

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        # filter examples that have longer lengths than limits
        if filter_func(example):
            continue

        total += 1
        para_idxs = np.zeros([para_limit], dtype=np.int32)
        para_idxs_unk = np.zeros([para_limit + 2], dtype=np.int32)
        para_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit + 2], dtype=np.int32)
        ques_idxs_unk = np.zeros([ques_limit + 2], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        labels = np.zeros([para_limit], dtype=np.int32)
        pos_tags = np.zeros([para_limit], dtype=np.int32)
        ner_tags = np.zeros([para_limit], dtype=np.int32)
        que_labels = np.zeros([ques_limit], dtype=np.int32)
        que_pos_tags = np.zeros([ques_limit], dtype=np.int32)
        que_ner_tags = np.zeros([ques_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        answer_start = example["answer_start"]
        answer_end = example["answer_end"]
        y1[answer_start] = 1.0
        y2[answer_end] = 1.0

        for i, token in enumerate(example["paragraph"]):
            wid = _get_id(token, word2idx_dict)
            # for unknown words, use their position indexes
            para_idxs[i] = len(word2idx_dict) + i if wid == 1 else wid
            # get char
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                para_char_idxs[i, j] = _get_id(char, char2idx_dict)

        for i, token in enumerate(example["paragraph_unk"]):
            para_idxs_unk[i] = _get_id(token, word2idx_dict)

        for i, token in enumerate(example["question"]):
            wid = _get_id(token, word2idx_dict)
            # for unknown words, use 1 or their position indexes in paragraph
            ques_idxs[i] = len(word2idx_dict) + example["paragraph"].index(token) \
                if wid == 1 and token in example["paragraph"] else wid
            ques_idxs_unk[i] = wid

        for i, token in enumerate(example["question"][1:-1]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_id(char, char2idx_dict)

        for i, (label, pos, ner) in enumerate(zip(example["labels"], example["pos_tags"], example["ner_tags"])):
            labels[i] = _get_id(label, label2idx_dict)
            pos_tags[i] = _get_id(pos, pos2idx_dict)
            ner_tags[i] = _get_id(ner, ner2idx_dict)

        for i, (label, pos, ner) in enumerate(zip(example["que_labels"], example["que_pos_tags"], example["que_ner_tags"])):
            que_labels[i] = _get_id(label, label2idx_dict)
            que_pos_tags[i] = _get_id(pos, pos2idx_dict)
            que_ner_tags[i] = _get_id(ner, ner2idx_dict)

        record = tf.train.Example(features=tf.train.Features(feature={
            "para_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[para_idxs.tostring()])),
            "para_idxs_unk": tf.train.Feature(bytes_list=tf.train.BytesList(value=[para_idxs_unk.tostring()])),
            "para_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[para_char_idxs.tostring()])),
            "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
            "ques_idxs_unk": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs_unk.tostring()])),
            "ques_char_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_idxs.tostring()])),
            "labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tostring()])),
            "pos_tags": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pos_tags.tostring()])),
            "ner_tags": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ner_tags.tostring()])),
            "que_labels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[que_labels.tostring()])),
            "que_pos_tags": tf.train.Feature(bytes_list=tf.train.BytesList(value=[que_pos_tags.tostring()])),
            "que_ner_tags": tf.train.Feature(bytes_list=tf.train.BytesList(value=[que_ner_tags.tostring()])),
            "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
            "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]]))
        }))
        writer.write(record.SerializeToString())
    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def prepare(config):
    # process files
    train_examples, train_eval = process_file(config.train_para_file, config.train_question_file,
                                              config.train_answer_file, lower_word=config.lower_word)
    dev_examples, dev_eval = process_file(config.dev_para_file, config.dev_question_file,
                                          config.dev_answer_file, lower_word=config.lower_word)
    test_examples, test_eval = process_file(config.test_para_file, config.test_question_file,
                                            config.test_answer_file, lower_word=config.lower_word)

    with open(config.word_dictionary, "r") as fh:
        word2idx_dict = json.load(fh)
    with open(config.char_dictionary, "r") as fh:
        char2idx_dict = json.load(fh)
    with open(config.label_dictionary, "r") as fh:
        label2idx_dict = json.load(fh)
    with open(config.pos_dictionary, "r") as fh:
        pos2idx_dict = json.load(fh)
    with open(config.ner_dictionary, "r") as fh:
        ner2idx_dict = json.load(fh)

    train_meta = build_features(config, train_examples, "train", config.train_record_file, word2idx_dict,
                                pos2idx_dict, ner2idx_dict, label2idx_dict, char2idx_dict)
    dev_meta = build_features(config, dev_examples, "dev", config.dev_record_file, word2idx_dict,
                              pos2idx_dict, ner2idx_dict, label2idx_dict, char2idx_dict)
    test_meta = build_features(config, test_examples, "test", config.test_record_file, word2idx_dict,
                               pos2idx_dict, ner2idx_dict, label2idx_dict, char2idx_dict, is_test=True)

    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.train_meta, train_meta, message="train meta")
    save(config.dev_meta, dev_meta, message="dev meta")
    save(config.test_meta, test_meta, message="test meta")