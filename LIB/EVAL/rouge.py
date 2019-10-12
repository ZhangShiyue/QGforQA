"""
Code is from https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/rouge/rouge.py
"""


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if (len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for _ in range(0, len(sub) + 1)] for _ in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


def compute_rouge_L(pred, refs, beta=1.2):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    prec = []
    rec = []
    for ref in refs:
        # compute the longest common subsequence
        lcs = my_lcs(pred, ref)
        prec.append(lcs / float(len(pred)) if len(pred) != 0 else 0.0)
        rec.append(lcs / float(len(ref)) if len(ref) != 0 else 0.0)

    prec_max = max(prec)
    rec_max = max(rec)

    if prec_max != 0 and rec_max != 0:
        score = ((1 + beta ** 2) * prec_max * rec_max) / float(rec_max + beta ** 2 * prec_max)
    else:
        score = 0.0
    return score