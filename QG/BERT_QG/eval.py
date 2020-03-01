import numpy as np
from collections import Counter
import sys
sys.path.append('../..')

from LIB.EVAL.meteor import Meteor
from LIB.EVAL.bleu import compute_bleu, diverse_bleu
from LIB.EVAL.rouge import compute_rouge_L


def convert_to_words(map_to_orig, question):
    """convert bert word pieces to normal tokens"""
    i = 0
    words = []
    while i < len(question):
        end = i + 1
        word = question[i]
        for j in range(i + 1, len(question)):
            if j - i > 17:
                break
            key = tuple(question[i:j])
            if key in map_to_orig:
                word = map_to_orig[key]
                end = j
        i = end
        words.append(word.replace("##", ""))
    return words


def convert_tokens_seq(eval_file, qa_id, symbols, probs, id2word, map_to_orig):
    "id sequence to token sequence"
    def _get_penalty(syms):
        trigrams = [tuple(syms[i: i + 3]) for i in range(len(syms) - 2)]
        repeat_trigram = list(filter(lambda x: x > 1, list(Counter(trigrams).values()))) != []
        return repeat_trigram

    answer_dict = {}
    for qid, prob, bsyms in zip(qa_id, probs, zip(*symbols)):
        answers = []
        try:
            bsyms = zip(*bsyms)
        except:
            bsyms = [bsyms]
        for p, syms in zip(prob, bsyms):
            context_tokens = eval_file[str(qid)]["paragraph"]
            if 102 in syms:
                syms = syms[:syms.index(102)]
            syms = [id2word[sym] if sym in id2word
                    else context_tokens[sym - len(id2word)] for sym in syms]
            tokens = convert_to_words(map_to_orig, syms)
            answer = u' '.join(tokens)
            lp, penalty = len(tokens) + 1, 0.
            if _get_penalty(tokens):
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
        ground_truths = [list(map(lambda x: x.lower(), eval_file[key]["question_eval"]))]
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
        ground_truths = [list(map(lambda x: x.lower(), eval_file[key]["question_eval"]))]
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

