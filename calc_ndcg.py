""" input format: key \t label \t score
    output: ndcg
"""
import sys, math


def DCG(label_list):
    dcgsum = 0
    for i in range(len(label_list)):
        dcg = (2 ** label_list[i] - 1) / math.log(i + 2, 2)
        dcgsum += dcg
    return dcgsum

def NDCG(label_list, topK):
    dcg = DCG(label_list[:topK])
    ideal_list = sorted(label_list, reverse=True)
    ideal_dcg = DCG(ideal_list[:topK]) # if the label is [1, 0, ..., 0], ideal_dcg = 1
    if ideal_dcg == 0:
        return 0
    return dcg / ideal_dcg

def queryNDCG(label_qid_score):
    tmp = sorted(label_qid_score, key = lambda x:-x[2])
    label_list = []
    for label,q,s in tmp:
        label_list.append(label)
    return NDCG(label_list, 1), NDCG(label_list, 3)
    
last_qid = ""
l_q_s = []

ndcg = [0, 0]
cnt = 0
positive_label = False

for line in sys.stdin:
    qid, label, score = line.rstrip().split("\t")
    if last_qid != "" and qid != last_qid:
        if positive_label:
            ndcg1, ndcg3 = queryNDCG(l_q_s)
            ndcg[0] += ndcg1
            ndcg[1] += ndcg3
            cnt += 1
        l_q_s= []
        positive_label = False
    if label == '1':
        positive_label = True
    last_qid = qid
    
    if len(l_q_s) < 10:
        l_q_s.append([int(label), qid, float(score)])

if last_qid != "" and positive_label:
    ndcg1, ndcg3 = queryNDCG(l_q_s)
    ndcg[0] += ndcg1
    ndcg[1] += ndcg3
    cnt += 1

print "ndcg@1: %f\tndcg@3: %f" % (ndcg[0]/cnt, ndcg[1]/cnt)
print "total:  %d\tndcg@1_num: %f\tndcg@3_num: %f" % (cnt, ndcg[0], ndcg[1])
