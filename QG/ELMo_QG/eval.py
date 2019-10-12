import numpy as np
from collections import Counter
import sys
sys.path.append('../..')

from LIB.EVAL.meteor import Meteor
from LIB.EVAL.bleu import compute_bleu, diverse_bleu
from LIB.EVAL.rouge import compute_rouge_L


def convert_tokens(eval_file, qa_id, symbols, probs, id2word):
    def _get_penalty(syms):
        bigrams = [tuple(syms[i: i + 2]) for i in range(len(syms) - 1)]
        repeat_bigram = list(filter(lambda x: x > 1, list(Counter(bigrams).values()))) != []
        trigrams = [tuple(syms[i: i + 3]) for i in range(len(syms) - 2)]
        repeat_trigram = list(filter(lambda x: x > 1, list(Counter(trigrams).values()))) != []
        return repeat_bigram or repeat_trigram

    answer_dict = {}
    for qid, prob, bsyms in zip(qa_id, probs, zip(*symbols)):
        answers = []
        try:
            bsyms = zip(*bsyms)
        except:
            bsyms = [bsyms]
        for p, syms in zip(prob, bsyms):
            context_tokens = eval_file[str(qid)]["paragraph"]
            if 3 in syms:
                syms = syms[:syms.index(3)]
            syms = [id2word[sym] if sym in id2word
                    else context_tokens[sym - len(id2word)] for sym in syms]
            answer = u' '.join(syms)
            lp, penalty = len(syms) + 1, 0.
            if _get_penalty(syms):
                penalty = 1.0
            answers.append((p / lp - penalty, answer))
        answer_dict[str(qid)] = answers
    return answer_dict


def evaluate(eval_file, answer_dict):
    reference_corpus = []
    translation_corpus = []
    translation_corpus_rouge_oracle = []
    translation_corpus_bleu_oracle = []
    rouges = []
    div_bleus = []
    rouges_oracle = []
    meteor = Meteor()
    res, res_oracle, gts = [], [], []
    for key, answers in answer_dict.items():
        answers = sorted(answers, key=lambda x: x[0], reverse=True)
        ground_truths = [list(map(lambda x: x.lower(), eval_file[key]["question"][1:-1]))]
        prediction = answers[0][1].lower().split()
        answers_tmp = []
        for i, answer in enumerate(answers):
            rouge = compute_rouge_L(answer[1].lower().split(), ground_truths)
            mete = meteor.compute_score([[' '.join(ground_truth) for ground_truth in ground_truths]],
                                        [' '.join(answer[1].lower().split())])
            bleu = compute_bleu([ground_truths], [answer[1].lower().split()], smooth=True)
            answers_tmp.append((rouge, mete[0], bleu[0], answer[0], answer[1]))
        answers_rouge = sorted(answers_tmp, key=lambda x: x[0], reverse=True)
        answers_mete = sorted(answers_tmp, key=lambda x: x[1], reverse=True)
        answers_bleu = sorted(answers_tmp, key=lambda x: x[2], reverse=True)
        prediction_rouge_oracle = answers_rouge[0][4].lower().split()
        prediction_mete_oracle = answers_mete[0][4].lower().split()
        prediction_bleu_oracle = answers_bleu[0][4].lower().split()
        translation_corpus.append(prediction)
        translation_corpus_rouge_oracle.append(prediction_rouge_oracle)
        translation_corpus_bleu_oracle.append(prediction_bleu_oracle)
        reference_corpus.append(ground_truths)
        rouge = compute_rouge_L(prediction, ground_truths)
        rouge_oracle = compute_rouge_L(prediction_rouge_oracle, ground_truths)
        rouges.append(rouge)
        rouges_oracle.append(rouge_oracle)
        res.append(' '.join(prediction))
        res_oracle.append(' '.join(prediction_mete_oracle))
        gts.append([' '.join(ground_truth) for ground_truth in ground_truths])
        div_bleus.append(diverse_bleu(answers))
    bleu = compute_bleu(reference_corpus, translation_corpus)
    bleu_oracle = compute_bleu(reference_corpus, translation_corpus_bleu_oracle)
    mete = meteor.compute_score(gts, res)
    mete_oracle = meteor.compute_score(gts, res_oracle)
    return {"bleu": bleu[0] * 100, "meteor": mete[0] * 100, "rougeL": np.mean(rouges) * 100,
            "bleu_oracle": bleu_oracle[0] * 100, "meteor_oracle": mete_oracle[0] * 100,
            "rougeL_oracle": np.mean(rouges_oracle) * 100, "diverse_bleu": np.mean(div_bleus) * 100}


def evaluate_simple(eval_file, answer_dict):
    reference_corpus = []
    translation_corpus = []
    rouges = []
    meteor = Meteor()
    res, gts = [], []
    for key, answers in answer_dict.items():
        answers = sorted(answers, key=lambda x: x[0], reverse=True)
        ground_truths = [list(map(lambda x: x.lower(), eval_file[key]["question"][1:-1]))]
        prediction = answers[0][1].lower().split()
        translation_corpus.append(prediction)
        reference_corpus.append(ground_truths)
        rouge = compute_rouge_L(prediction, ground_truths)
        rouges.append(rouge)
        res.append(' '.join(prediction))
        gts.append([' '.join(ground_truth) for ground_truth in ground_truths])
    bleu = compute_bleu(reference_corpus, translation_corpus)
    mete = meteor.compute_score(gts, res)
    return {"bleu": bleu[0] * 100, "meteor": mete[0] * 100, "rougeL": np.mean(rouges) * 100}


def evaluate_rl(eval_file, qa_id, symbols, symbols_rl, id2word, metric="rouge"):
    meteor = Meteor()
    ques_limit = len(symbols)
    batch_size, _ = symbols[0].shape
    rewards = np.zeros([batch_size], dtype=np.float32)
    rewards_rl = np.zeros([batch_size], dtype=np.float32)
    rewards_base = np.zeros([batch_size], dtype=np.float32)
    ques_rl = np.zeros([batch_size, ques_limit], dtype=np.int32)
    for i, (qid, syms, syms_rl) in enumerate(zip(qa_id, zip(*symbols), zip(*symbols_rl))):
        syms = list(np.reshape(syms, [-1]))
        syms_rl = list(syms_rl)
        ground_truths = [list(map(lambda x: x.lower(), eval_file[str(qid)]["question"][1:-1]))]
        context_tokens = eval_file[str(qid)]["paragraph"]
        if 3 in syms:
            syms = syms[:syms.index(3)]
        if 3 in syms_rl:
            syms_rl = syms_rl[:syms_rl.index(3)]
        prediction = [id2word[sym] if sym in id2word
                      else context_tokens[sym - len(id2word)] for sym in syms]
        prediction_rl = [id2word[sym] if sym in id2word
                         else context_tokens[sym - len(id2word)] for sym in syms_rl]
        if metric == "rouge":
            reward_base = compute_rouge_L(prediction, ground_truths)
            reward_rl = compute_rouge_L(prediction_rl, ground_truths)
        elif metric == "bleu":
            reward_base, _ = compute_bleu([ground_truths], [prediction], smooth=True)
            reward_rl, _ = compute_bleu([ground_truths], [prediction_rl], smooth=True)
        elif metric == "meteor":
            reward_base = meteor.compute_score([[' '.join(ground_truth) for ground_truth in ground_truths]],
                                                [' '.join(prediction)])[0]
            reward_rl = meteor.compute_score([[' '.join(ground_truth) for ground_truth in ground_truths]],
                                                [' '.join(prediction_rl)])[0]
        else:
            print("Wrong Metric!")
            exit()
        rewards[i] = reward_rl - reward_base
        rewards_rl[i] = reward_rl
        rewards_base[i] = reward_base
        # add GO and EOS
        syms_rl = [2] + syms_rl[:ques_limit-2] + [3]
        for j, sym_rl in enumerate(syms_rl):
            ques_rl[i, j] = sym_rl
    return rewards, rewards_rl, rewards_base, ques_rl


def format_generated_ques_for_qpc(qa_id, symbols, symbols_rl, batch_size, ques_limit, id2word):
    que, que_rl = np.zeros([batch_size, ques_limit + 2], dtype=np.int32), \
                  np.zeros([batch_size, ques_limit + 2], dtype=np.int32)
    que_unk, que_unk_rl = np.zeros([batch_size, ques_limit + 2], dtype=np.int32), \
                          np.zeros([batch_size, ques_limit + 2], dtype=np.int32)
    for k, (qid, syms, syms_rl) in enumerate(zip(qa_id, zip(*symbols), zip(*symbols_rl))):
        syms = list(np.reshape(syms, [-1]))
        syms_rl = list(syms_rl)
        syms, syms_rl = list(syms), list(syms_rl)
        if 3 in syms:
            syms = syms[:syms.index(3)]
        syms = [2] + syms[:ques_limit] + [3]
        for i, sym in enumerate(syms):
            que[k, i] = sym
            que_unk[k, i] = sym if sym in id2word else 1
        if 3 in syms_rl:
            syms_rl = syms_rl[:syms_rl.index(3)]
        syms_rl = [2] + syms_rl[:ques_limit] + [3]
        for i, sym_rl in enumerate(syms_rl):
            que_rl[k, i] = sym_rl
            que_unk_rl[k, i] = sym_rl if sym_rl in id2word else 1
    return que, que_unk, que_rl, que_unk_rl


def format_generated_ques_for_qa(eval_file, qa_id, symbols, symbols_rl, batch_size, ques_limit,
                                 char_limit, id2word, char2idx_dict):
    que, que_rl = np.zeros([batch_size, ques_limit + 2], dtype=np.int32), \
                  np.zeros([batch_size, ques_limit + 2], dtype=np.int32)
    que_unk, que_unk_rl = np.zeros([batch_size, ques_limit + 2], dtype=np.int32), \
                          np.zeros([batch_size, ques_limit + 2], dtype=np.int32)
    que_char, que_char_rl = np.zeros([batch_size, ques_limit, char_limit], dtype=np.int32), \
                            np.zeros([batch_size, ques_limit, char_limit], dtype=np.int32)
    for k, (qid, syms, syms_rl) in enumerate(zip(qa_id, zip(*symbols), zip(*symbols_rl))):
        syms = list(np.reshape(syms, [-1]))
        syms_rl = list(syms_rl)
        para_tokens = eval_file[str(qid)]["paragraph"]
        syms, syms_rl = list(syms), list(syms_rl)
        if 3 in syms:
            syms = syms[:syms.index(3)]
        syms = [2] + syms[:ques_limit] + [3]
        for i, sym in enumerate(syms):
            que[k, i] = sym
            que_unk[k, i] = sym if sym in id2word else 1
            word = id2word[sym] if sym in id2word else para_tokens[sym - len(id2word)]
            if i == 0 or i == ques_limit + 1:
                continue
            for j, c in enumerate(list(word)):
                if j == char_limit:
                    break
                que_char[k, i-1, j] = char2idx_dict[c] if c in char2idx_dict else 1
        if 3 in syms_rl:
            syms_rl = syms_rl[:syms_rl.index(3)]
        syms_rl = [2] + syms_rl[:ques_limit] + [3]
        for i, sym_rl in enumerate(syms_rl):
            que_rl[k, i] = sym_rl
            que_unk_rl[k, i] = sym_rl if sym_rl in id2word else 1
            word = id2word[sym_rl] if sym_rl in id2word else para_tokens[sym_rl - len(id2word)]
            if i == 0 or i == ques_limit + 1:
                continue
            for j, c in enumerate(list(word)):
                if j == char_limit:
                    break
                que_char_rl[k, i-1, j] = char2idx_dict[c] if c in char2idx_dict else 1
    return que, que_unk, que_char, que_rl, que_unk_rl, que_char_rl
