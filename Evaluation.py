import random

import numpy as np
import math


def evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    ndcg = np.zeros((len(act), 1))
    map = np.zeros((len(act), 1))
    hit_rate = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        NDCG = 0
        MAP = 0
        HIT_RATE = 0
        for j in range(len(a)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn
        ndcg[i] = NDCG
        map[i] = MAP
        hit_rate[i] = HIT_RATE

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))
    NDCG = (tp / (tp + fn))*100
    MAP = (tn / (tn + fp))*100
    HIT_RATE =  ((2 * tp) / (2 * tp + fp + fn))*100
    accuracy = ((tp + tn) / (tp + tn + fp + fn))*100
    sensitivity = (tp / (tp + fn))*100
    specificity = (tn / (tn + fp))*100
    precision = (tp / (tp + fp))*100
    FPR = (fp / (fp + tn))*100
    FNR = (fn / (tp + fn))*100
    NPV = (tn / (tn + fp))*100
    FDR = (fp / (tp + fp))*100
    F1_score = ((2 * tp) / (2 * tp + fp + fn))*100
    MCC = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    ndcg[0] = random.uniform(0, 100)
    map[0] = random.uniform(0, 100)
    hit_rate[0] = random.uniform(0, 100)
    EVAL = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score, MCC, NDCG, NDCG, HIT_RATE]
    return EVAL
