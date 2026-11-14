# CIDEr (Vedantam et al., https://arxiv.org/abs/1411.5726), the Cider class of pycocoevalcap
# (https://github.com/tylin/coco-caption, BSD 2-clause, Tsung-Yi Lin and Ramakrishna Vedantam) in one function,
# with the same numbers. Sentences are split on whitespace, without any tokenizer.
from collections import Counter

import numpy as np


def _ngrams(sentence, n):
    """Counts of every 1..n-gram of the sentence."""
    words = sentence.split()
    counts = Counter()
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            counts[tuple(words[i:i + k])] += 1
    return counts


def cider(
    gts,          # {id: [reference sentence, ...]}
    res,          # {id: [hypothesis sentence]}, same ids
    n=4,          # longest n-gram
    sigma=6.0,    # std of the Gaussian length penalty
):
    """Mean CIDEr over the ids (x10 as pycocoevalcap, multiply by 100 for the usual numbers)."""
    refs = [[_ngrams(r, n) for r in gts[k]] for k in gts]
    hyps = [_ngrams(res[k][0], n) for k in gts]
    # document frequency: in how many ids' references each n-gram appears
    df = Counter(g for rs in refs for g in {g for r in rs for g in r})
    ref_len = np.log(float(len(refs)))

    def tfidf(counts):
        vec, norm, length = [{} for _ in range(n)], [0.0] * n, 0
        for g, tf in counts.items():
            i = len(g) - 1
            vec[i][g] = float(tf) * (ref_len - np.log(max(1.0, df[g])))
            norm[i] += pow(vec[i][g], 2)
            if i == 1:   # the length counts bigrams, as in pycocoevalcap
                length += tf
        return vec, [np.sqrt(x) for x in norm], length

    scores = []
    for h, rs in zip(hyps, refs):
        vh, nh, lh = tfidf(h)
        score = np.zeros(n)
        for r in rs:
            vr, nr, lr = tfidf(r)
            val = np.zeros(n)
            for i in range(n):
                for g, x in vh[i].items():
                    y = vr[i].get(g, 0.0)
                    val[i] += min(x, y) * y   # clipped
                if nh[i] != 0 and nr[i] != 0:
                    val[i] /= nh[i] * nr[i]
                val[i] *= np.e ** (-(float(lh - lr) ** 2) / (2 * sigma ** 2))
            score += val
        scores.append(np.mean(score) / len(rs) * 10.0)
    return float(np.mean(np.array(scores)))
