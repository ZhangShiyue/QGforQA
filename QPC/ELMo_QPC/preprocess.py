import os
import corenlp


def process(orig_file, output_dir):
    print("Generating {} raw data...".format(orig_file))
    max_que_len = 0
    qqp_file = "{}/qqp.tok".format(output_dir)
    with open(orig_file, "r") as fh, corenlp.CoreNLPClient(annotators="tokenize ssplit".split(),
                                                          endpoint="http://localhost:9099", timeout=50000) as client:
        fh.readline()
        res = []
        while True:
            line = fh.readline()
            if not line:
                break
            try:
                id, qid1, qid2, question1, question2, label = line.strip().split('\t')
                ann_que1 = client.annotate(question1)
                question1 = [token.word for sent in ann_que1.sentence for token in sent.token]
                max_que_len = max(max_que_len, len(question1))
                ann_que2 = client.annotate(question2)
                question2 = [token.word for sent in ann_que2.sentence for token in sent.token]
                max_que_len = max(max_que_len, len(question2))
                label = int(label)
                res.append("{}\t{}\t{}".format(' '.join(question1), ' '.join(question2), label))
            except:
                continue
        print(max_que_len)
        with open(qqp_file, 'w') as f:
            f.write('\n'.join(res))


def get_data(train_file, dev_file, output_dir):
    # process(train_file, "{}/train/".format(output_dir))
    process(dev_file, "{}/dev/".format(output_dir))


if __name__ == '__main__':
    # process data
    os.system("mkdir data; mkdir data/processed; mkdir data/processed/train; mkdir data/processed/dev")
    get_data("../../LIB/qqp/QQP/train.tsv", "../../LIB/qqp/QQP/dev.tsv", "data/processed")
