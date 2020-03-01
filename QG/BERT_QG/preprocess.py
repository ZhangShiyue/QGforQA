import os
import sys
sys.path.append('../..')
import pickle as pkl
from LIB.bert import tokenization


def bert_tokenize():
    map_to_orig = {}
    tokenizer = tokenization.FullTokenizer(vocab_file="../../LIB/bert/uncased_L-12_H-768_A-12/vocab.txt",
                                           do_lower_case=True)
    for dir in ["train", "dev", "test"]:
        with open("data/processed/{}/paras.label".format(dir), 'r') as fpl, open("data/processed/{}/paras.tok".format(dir), 'r') as fp, \
                open("data/processed/{}/paras.pos".format(dir), 'r') as fpp, open("data/processed/{}/paras.ner".format(dir), 'r') as fpn:
            while True:
                label = fpl.readline()
                para = fp.readline().lower()
                pos = fpp.readline()
                ner = fpn.readline()
                if not para:
                    break
                para_tokens = para.strip().split(' ')
                pos_tags = pos.strip().split(' ')
                ner_tags = ner.strip().split(' ')
                labels = label.strip().split(' ')
                new_para_tokens, new_pos_tags, new_ner_tags, new_labels = [], [], [], []
                for w, p, n, l in zip(para_tokens, pos_tags, ner_tags, labels):
                    nws = tokenizer.tokenize(w, unk=False)
                    map_to_orig[tuple(nws)] = w
                    new_para_tokens.extend(nws)
                    new_pos_tags.extend([p] * len(nws))
                    new_ner_tags.extend([n] * len(nws))
                    new_labels.extend([l] * len(nws))
                assert len(new_para_tokens) == len(new_pos_tags) == len(new_ner_tags) == len(new_labels)

                with open("data/processed_bert/{}/paras.label".format(dir), 'a') as f:
                    f.write(' '.join(new_labels) + '\n')
                with open("data/processed_bert/{}/paras.tok".format(dir), 'a') as f:
                    f.write(' '.join(new_para_tokens) + '\n')
                with open("data/processed_bert/{}/paras.pos".format(dir), 'a') as f:
                    f.write(' '.join(new_pos_tags) + '\n')
                with open("data/processed_bert/{}/paras.ner".format(dir), 'a') as f:
                    f.write(' '.join(new_ner_tags) + '\n')

        with open("data/processed/{}/questions.tok".format(dir), 'r') as fq, \
                open("data/processed/{}/questions.pos".format(dir), 'r') as fqp, \
                open("data/processed/{}/questions.ner".format(dir), 'r') as fqn:
            while True:
                que = fq.readline().lower()
                pos = fqp.readline()
                ner = fqn.readline()
                if not que:
                    break
                que_tokens = que.strip().split(' ')
                pos_tags = pos.strip().split(' ')
                ner_tags = ner.strip().split(' ')
                new_que_tokens, new_pos_tags, new_ner_tags = [], [], []
                for w, p, n in zip(que_tokens, pos_tags, ner_tags):
                    nws = tokenizer.tokenize(w, unk=False)
                    map_to_orig[tuple(nws)] = w
                    new_que_tokens.extend(nws)
                    new_pos_tags.extend([p] * len(nws))
                    new_ner_tags.extend([n] * len(nws))
                assert len(new_que_tokens) == len(new_pos_tags) == len(new_ner_tags)

                with open("data/processed_bert/{}/questions.tok".format(dir), 'a') as f:
                    f.write(' '.join(new_que_tokens) + '\n')
                with open("data/processed_bert/{}/questions.pos".format(dir), 'a') as f:
                    f.write(' '.join(new_pos_tags) + '\n')
                with open("data/processed_bert/{}/questions.ner".format(dir), 'a') as f:
                    f.write(' '.join(new_ner_tags) + '\n')
                with open("data/processed_bert/{}/questions_eval.tok".format(dir), 'a') as f:
                    f.write(' '.join(que_tokens) + '\n')
    print(len(map_to_orig))
    with open("data/vocab/map_to_orig.pkl", 'wb') as f:
        pkl.dump(map_to_orig, f)


if __name__ == '__main__':
    # tokenize data with bert tokenizer
    os.system("mkdir data; mkdir data/vocab; mkdir data/processed_bert; mkdir data/processed_bert/train; "
              "mkdir data/processed_bert/dev; mkdir data/processed_bert/test")
    bert_tokenize()
