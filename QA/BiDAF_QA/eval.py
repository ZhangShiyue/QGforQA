import sys
sys.path.append('../..')

from LIB.EVAL.evaluate import metric_max_over_ground_truths, f1_score, exact_match_score


def convert_tokens_qa_for_qg(eval_file, qa_id, pp1, pp2, bprobs):
    answer_dict = {}
    for qid, bp1, bp2, bprob in zip(qa_id, pp1, pp2, bprobs):
        answers = []
        context_tokens = eval_file[str(qid)]["paragraph"]
        for p1, p2, bp in zip(bp1, bp2, bprob):
            answer = context_tokens[p1: p2 + 1]
            answers.append((p1, p2, bp, answer))
        answer_dict[str(qid)] = answers
    return answer_dict


def evaluate_qa_for_qg(eval_file, answer_dict):
    f1 = exact_match = total = 0
    oracle_f1 = oracle_em = 0
    for key, values in answer_dict.items():
        values = sorted(values, key=lambda x: x[2])
        total += 1
        ground_truths = [' '.join(answer) for answer in eval_file[key]["answer"]]
        answers_tmp = []
        for value in values:
            prediction = ' '.join(value[3])
            f1_tmp = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
            em_tmp = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
            answers_tmp.append((f1_tmp, em_tmp, value[0], value[1], value[2], value[3]))
        f1 += answers_tmp[0][0]
        exact_match += answers_tmp[0][1]
        oracle_f1 += max(map(lambda x: x[0], answers_tmp))
        oracle_em += max(map(lambda x: x[1], answers_tmp))
    exact_match = 100.0 * exact_match / total
    oracle_em = 100.0 * oracle_em / total
    f1 = 100.0 * f1 / total
    oracle_f1 = 100.0 * oracle_f1 / total
    return {'em': exact_match, 'f1': f1, "oracle_em": oracle_em, "oracle_f1": oracle_f1}
