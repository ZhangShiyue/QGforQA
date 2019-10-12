"""
Data pre-processing
"""
import os
import corenlp
import numpy as np
import ujson as json
from tqdm import tqdm
from collections import Counter
from bilm import dump_token_embeddings
import sys
sys.path.append('../..')

from LIB.utils import save


def process(json_file, outpur_dir, exclude_titles=None, include_titles=None):
    """
    :param json_file: original data in json format
    :param outpur_dir: the output directory of pre-processed data
    :param exclude_titles: article titles to exclude
    :param include_titles: article titles to include
    """
    para_file = "{}/paras".format(outpur_dir)
    question_file = "{}/questions".format(outpur_dir)
    sent_file = "{}/sents".format(outpur_dir)
    answer_file = "{}/answers".format(outpur_dir)
    print("Generating {} raw data...".format(json_file))
    max_sent, max_sent_len, max_que_len, max_ans_len = 0, 0, 0, 0
    with open(json_file, "r") as fh, corenlp.CoreNLPClient(annotators="tokenize ssplit pos ner".split(),
                                                           endpoint="http://localhost:9099", timeout=50000) as client:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            title = article["title"]
            if include_titles and title not in include_titles:
                continue
            if exclude_titles and title in exclude_titles:
                continue
            for para in article["paragraphs"]:
                paragraphs, questions, answers, sents, ids = [], [], [], [], []
                paragraphs_pos, questions_pos, answers_pos, sents_pos = [], [], [], []
                paragraphs_ner, questions_ner, answers_ner, sents_ner = [], [], [], []
                answers_index, sents_index = [], []
                # paragraph
                context = para["context"]
                if not context.strip():
                    continue
                ann_para = client.annotate(context)
                max_sent = max(max_sent, len(ann_para.sentence))
                max_sent_len = max(max_sent_len, max(map(lambda x: len(x.token), ann_para.sentence)))
                ann_para_tokens, paragraph_tokens, paragraph_pos, paragraph_ner = [], [], [], []
                for sent in ann_para.sentence:
                    for token in sent.token:
                        ann_para_tokens.append(token)
                        paragraph_tokens.append(token.word)
                        paragraph_pos.append(token.pos)
                        paragraph_ner.append(token.ner)

                # questions
                for qa in para["qas"]:
                    # question
                    ques = qa["question"]
                    id = qa["id"]
                    if not ques.strip():
                        continue
                    ann_que = client.annotate(ques)
                    max_que_len = max(max_que_len, len(ann_que.sentence[0].token))
                    question_tokens, question_pos, question_ner = [], [], []
                    for sent in ann_que.sentence:
                        for token in sent.token:
                            question_tokens.append(token.word)
                            question_pos.append(token.pos)
                            question_ner.append(token.ner)

                    # answer
                    all_answer_tokens, all_answer_pos, all_answer_ner, all_answer_index = [], [], [], []
                    all_sent_tokens, all_sent_pos, all_sent_ner, all_sent_index = [], [], [], []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        if not answer_text.strip():
                            continue
                        ann_ans = client.annotate(answer_text)
                        answer_tokens, answer_pos, answer_ner = [], [], []
                        for sent in ann_ans.sentence:
                            for token in sent.token:
                                answer_tokens.append(token.word)
                                answer_pos.append(token.pos)
                                answer_ner.append(token.ner)
                        all_answer_tokens.append(' '.join(answer_tokens))
                        all_answer_pos.append(' '.join(answer_pos))
                        all_answer_ner.append(' '.join(answer_ner))

                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        # sentence
                        sentence = []
                        for sent in ann_para.sentence:
                            if sent.characterOffsetBegin <= answer_start <= sent.characterOffsetEnd or \
                                    sent.characterOffsetBegin <= answer_end <= sent.characterOffsetEnd:
                                sentence.append(sent)
                        sentence = [token for sent in sentence for token in sent.token]
                        sentence_tokens = [token.word for token in sentence]
                        sentence_pos = [token.pos for token in sentence]
                        sentence_ner = [token.ner for token in sentence]
                        all_sent_tokens.append(' '.join(sentence_tokens))
                        all_sent_pos.append(' '.join(sentence_pos))
                        all_sent_ner.append(' '.join(sentence_ner))

                        # sentence index
                        y1_sent = sentence[0].tokenBeginIndex
                        y2_sent = sentence[-1].tokenBeginIndex
                        # answer index
                        y1_ans = None
                        for i, token in enumerate(sentence):
                            if token.beginChar - 1 <= answer_start <= token.endChar:
                                y1_ans = sentence[0].tokenBeginIndex + i
                        try:
                            assert y1_ans != None
                        except:
                            continue
                        y2_ans = y1_ans + len(answer_tokens) - 1
                        all_answer_index.append("{},{}".format(y1_ans, y2_ans))
                        all_sent_index.append("{},{}".format(y1_sent, y2_sent))

                    paragraphs.append(' '.join(paragraph_tokens))
                    paragraphs_pos.append(' '.join(paragraph_pos))
                    paragraphs_ner.append(' '.join(paragraph_ner))
                    questions.append(' '.join(question_tokens))
                    questions_pos.append(' '.join(question_pos))
                    questions_ner.append(' '.join(question_ner))
                    answers.append('\t'.join(all_answer_tokens))
                    answers_pos.append('\t'.join(all_answer_pos))
                    answers_ner.append('\t'.join(all_answer_ner))
                    answers_index.append('\t'.join(all_answer_index))
                    sents.append('\t'.join(all_sent_tokens))
                    sents_pos.append('\t'.join(all_sent_pos))
                    sents_ner.append('\t'.join(all_sent_ner))
                    sents_index.append('\t'.join(all_sent_index))
                    ids.append(id)

                # save para
                with open("{}.tok".format(para_file), 'a') as f:
                    f.write('\n'.join(paragraphs) + '\n')
                with open("{}.pos".format(para_file), 'a') as f:
                    f.write('\n'.join(paragraphs_pos) + '\n')
                with open("{}.ner".format(para_file), 'a') as f:
                    f.write('\n'.join(paragraphs_ner) + '\n')
                with open("{}.id".format(para_file), 'a') as f:
                    f.write('\n'.join(ids) + '\n')
                # save question
                with open("{}.tok".format(question_file), 'a') as f:
                    f.write('\n'.join(questions) + '\n')
                with open("{}.pos".format(question_file), 'a') as f:
                    f.write('\n'.join(questions_pos) + '\n')
                with open("{}.ner".format(question_file), 'a') as f:
                    f.write('\n'.join(questions_ner) + '\n')

                # save answer
                with open("{}.tok".format(answer_file), 'a') as f:
                    f.write('\n'.join(answers) + '\n')
                with open("{}.pos".format(answer_file), 'a') as f:
                    f.write('\n'.join(answers_pos) + '\n')
                with open("{}.ner".format(answer_file), 'a') as f:
                    f.write('\n'.join(answers_ner) + '\n')
                with open("{}.index".format(answer_file), 'a') as f:
                    f.write("\n".join(answers_index) + '\n')

                # save sent
                with open("{}.tok".format(sent_file), 'a') as f:
                    f.write('\n'.join(sents) + '\n')
                with open("{}.pos".format(sent_file), 'a') as f:
                    f.write('\n'.join(sents_pos) + '\n')
                with open("{}.ner".format(sent_file), 'a') as f:
                    f.write('\n'.join(sents_ner) + '\n')
                with open("{}.index".format(sent_file), 'a') as f:
                    f.write("\n".join(sents_index) + '\n')
    # get BIO labels
    label(para_file, answer_file)


def label(para_file, answer_file):
    # get the answer BIO label for paragraph
    max_node = 0
    with open("{}.tok".format(para_file), 'r') as fp, open("{}.label".format(para_file), 'a') as fl, \
            open("{}.index".format(answer_file), 'r') as fa:
        while True:
            para = fp.readline()
            if not para:
                break
            words = [p for p in para.strip().split(' ')]
            max_node = max(len(words), max_node)
            answer = fa.readline()
            labels = []
            try:
                start, end = map(int, answer.split('\t')[0].split(','))
                for i in range(len(words)):
                    if start <= i <= end:
                        # answer words
                        if i == start:
                            labels.append('B')
                        else:
                            labels.append('I')
                    else:
                        # non answer words
                        labels.append('O')
            except:
                pass
            fl.write(' '.join(labels) + '\n')
    return max_node


def get_data(train_json, dev_json, test_title_file, output_dir):
    test_titles = open(test_title_file, 'r').readlines()
    test_titles = set([line.strip() for line in test_titles])

    process(train_json, "{}/train/".format(output_dir), exclude_titles=test_titles)
    process(dev_json, "{}/dev/".format(output_dir))
    process(train_json, "{}/test/".format(output_dir), include_titles=test_titles)


def get_word_embedding(counter, emb_file, emb_size, vocab_size, vec_size, vocab_file):
    """
    get word embedding matrix from glove
    """
    print("Generating word embedding...")
    # load word embeddings
    embedding_dict = {}
    with open(emb_file, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, total=emb_size):
            array = line.split()
            word = "".join(array[0:-vec_size])
            vector = list(map(float, array[-vec_size:]))
            embedding_dict[word] = vector

    TRANSLATE = {
        "-lsb-": "[", "-rsb-": "]", "-lrb-": "(", "-rrb-": ")", "-lcb-": "{",
        "-rcb-": "}", "-LSB-": "[", "-RSB-": "]", "-LRB-": "(", "-RRB-": ")",
        "-LCB-": "{", "-RCB-": "}"
    }
    SPECIAL_TOKENS = ["<NULL>", "<UNK>", "<S>", "</S>"]
    words = list(map(lambda x: x[0], sorted(counter.items(), key=lambda x: x[1], reverse=True)))
    words = SPECIAL_TOKENS + words
    if vocab_size > 0:
        words = words[:vocab_size]
    with open(vocab_file, 'w') as f:
        f.write('\n'.join(words[1:]))
    embedding = np.random.normal(scale=0.1, size=(len(words), vec_size))
    word2idx_dict = {}
    unknown_count = 0
    for i, word in enumerate(words):
        word2idx_dict[word] = i
        if word in TRANSLATE:
            word = TRANSLATE[word]
        done = False
        for w in (word, word.lower(), word.upper(), word.capitalize()):
            if w in embedding_dict:
                embedding[i] = embedding_dict[w]
                done = True
                break
        if not done:
            unknown_count += 1
    return embedding, word2idx_dict, unknown_count


def get_tag_embedding(counter, data_type, vec_size):
    """
    get pos/ner/label tags' embedding matrix
    """
    print("Generating {} tag embedding...".format(data_type))
    SPECIAL_TOKENS = ["<NULL>", "<UNK>"]
    tags = list(map(lambda x: x[0], sorted(counter.items(), key=lambda x: x[1], reverse=True)))
    tags = SPECIAL_TOKENS + tags
    embedding = np.random.normal(scale=0.1, size=(len(tags), vec_size))
    word2idx_dict = {w: i for i, w in enumerate(tags)}
    return embedding, word2idx_dict


def get_vocab(config):
    print("Get the vocabulary...")
    word_counter, char_counter = Counter(), Counter()
    pos_counter, ner_counter, label_counter = Counter(), Counter(), Counter()
    files = [(config.train_para_file, config.train_question_file), (config.dev_para_file, config.dev_question_file)]
    for para_file, que_file in files:
        with open("{}.tok".format(para_file), 'r') as fp, open("{}.tok".format(que_file), 'r') as fq, \
                open("{}.pos".format(para_file), 'r') as fpp, open("{}.pos".format(que_file), 'r') as fqp, \
                open("{}.ner".format(para_file), 'r') as fpn, open("{}.ner".format(que_file), 'r') as fqn, \
                open("{}.label".format(para_file), 'r') as fpl:
            while True:
                para, question = fp.readline(), fq.readline()
                pos, que_pos = fpp.readline(), fqp.readline()
                ner, que_ner = fpn.readline(), fqn.readline()
                label = fpl.readline()
                if not question or not para:
                    break
                if config.lower_word:
                    para = para.lower()
                    question = question.lower()
                para_tokens = para.strip().split(' ')
                que_tokens = question.strip().split(' ')
                pos_tags = pos.strip().split(' ')
                ner_tags = ner.strip().split(' ')
                que_pos_tags = que_pos.strip().split(' ')
                que_ner_tags = que_ner.strip().split(' ')
                labels = label.strip().split(' ')
                for token in para_tokens + que_tokens:
                    word_counter[token] += 1
                    for char in list(token):
                        char_counter[char] += 1
                for pos_tag in pos_tags + que_pos_tags:
                    pos_counter[pos_tag] += 1
                for ner_tag in ner_tags + que_ner_tags:
                    ner_counter[ner_tag] += 1
                for label in labels:
                    label_counter[label] += 1
    word_emb_mat, word2idx_dict, unk_num = get_word_embedding(word_counter, emb_file=config.glove_word_file,
                                                              emb_size=config.glove_word_size,
                                                              vocab_size=config.vocab_size_limit,
                                                              vec_size=config.glove_dim, vocab_file=config.vocab_file)
    char_emb_mat, char2idx_dict = get_tag_embedding(char_counter, "char", vec_size=config.char_dim)
    pos_emb_mat, pos2idx_dict = get_tag_embedding(pos_counter, "pos", vec_size=config.pos_dim)
    ner_emb_mat, ner2idx_dict = get_tag_embedding(ner_counter, "ner", vec_size=config.ner_dim)
    label_emb_mat, label2idx_dict = get_tag_embedding(label_counter, "label", vec_size=config.label_dim)
    print("{} out of {} are not in glove".format(unk_num, len(word2idx_dict)))
    print("{} chars".format(char_emb_mat.shape[0]))
    print("{} pos tags, {} ner tags, {} answer labels, {} chars".format(
        pos_emb_mat.shape[0], ner_emb_mat.shape[0], label_emb_mat.shape[0], char_emb_mat.shape[0]))
    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.char_emb_file, char_emb_mat, message="char embedding")
    save(config.pos_emb_file, pos_emb_mat, message="pos embedding")
    save(config.ner_emb_file, ner_emb_mat, message="ner embedding")
    save(config.label_emb_file, label_emb_mat, message="label embedding")
    save(config.word_dictionary, word2idx_dict, message="word dictionary")
    save(config.char_dictionary, char2idx_dict, message="char dictionary")
    save(config.pos_dictionary, pos2idx_dict, message="pos dictionary")
    save(config.ner_dictionary, ner2idx_dict, message="ner dictionary")
    save(config.label_dictionary, label2idx_dict, message="label dictionary")
    print("Dump elmo word embedding...")
    token_embedding_file = config.embedding_file
    dump_token_embeddings(
        config.vocab_file, config.elmo_options_file, config.elmo_weight_file, token_embedding_file
    )


if __name__ == '__main__':
    # process data
    os.system("mkdir data; mkdir data/processed; mkdir data/processed/train; "
              "mkdir data/processed/dev; mkdir data/processed/test")
    get_data("../../LIB/squad/train-v1.1.json", "../../LIB/squad/dev-v1.1.json",
             "../../LIB/squad/doclist-test.txt", "data/processed")