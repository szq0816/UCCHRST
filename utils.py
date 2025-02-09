import torch
from os import path as osp
import numpy as np
def save_checkpoints(self):
    if self.task ==0:
        file_name = self.dataset + '_fusion.pth'
        ckp_path = osp.join(self.model_dir, 'real', file_name)
        obj = {
        'FusionTransformer': self.FuseTrans.state_dict()
        }
    if self.task ==1:
        file_name = self.dataset + '_hash_' + str(self.nbits)+".pth"
        ckp_path = osp.join(self.model_dir, 'hash', file_name)
        obj = {
        'FusionTransformer': self.FuseTrans.state_dict(),
        'ImageMlp': self.ImageMlp.state_dict(),
        'TextMlp': self.TextMlp.state_dict()
        }
    torch.save(obj, ckp_path)
    print('**********Save the {0} model successfully.**********'.format("real" if self.task==0 else "hash"))


def load_checkpoints(self, file_name):
    ckp_path = file_name
    try:
        obj = torch.load(ckp_path, map_location= self.device)
        print('**************** Load checkpoint %s ****************' % ckp_path)
    except IOError:
        print('********** Fail to load checkpoint %s!*********' % ckp_path)
        raise IOError
    if self.task==2:
        self.FuseTrans.load_state_dict(obj['FusionTransformer'])
    elif self.task==3:
        self.FuseTrans.load_state_dict(obj['FusionTransformer'])
        self.ImageMlp.load_state_dict(obj['ImageMlp'])
        self.TextMlp.load_state_dict(obj['TextMlp'])


def topk(cateTrainTest, IX, topk=5000):
    m, n = cateTrainTest.shape
    cateTrainTest = np.ascontiguousarray(cateTrainTest, np.int32).reshape(m * n)
    IX = np.ascontiguousarray(IX, np.int32).reshape(m * n)

    precs = np.zeros(n, dtype=np.float64)
    recs = np.zeros(n, dtype=np.float64)

    if topk == None:
        topk = m

    for i in range(n):
        retrieved_rel = 0
        for j in range(topk):
            idx = IX[i + n * j]
            retrieved_rel += cateTrainTest[i + n * idx]
        real_rel = 0
        for j in range(m):
            real_rel += cateTrainTest[i + n * j]

        precs[i] = retrieved_rel / (topk * 1.0)
        if real_rel != 0:
            recs[i] = retrieved_rel / (real_rel * 1.0)
        else:
            recs[i] = 0

    return np.mean(precs), np.mean(recs)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def getTrue2(test_label: np.ndarray, train_label: np.ndarray):
    cateTrainTest = np.sign(np.matmul(train_label, test_label.T))  # 0 or 1
    cateTrainTest = cateTrainTest.astype('int16')
    return cateTrainTest


def getHammingRank1(re_BT, qu_BI):
    HammTrainTesti2t = CalcHammingDist(re_BT, qu_BI)
    HammingRank1 = np.argsort(HammTrainTesti2t, axis=0)
    return HammingRank1


def cal_topK(qu_BI, re_BT, test_label, train_label, top_k):
    cateTrainTest = getTrue2(test_label, train_label)
    HammingRank1 = getHammingRank1(re_BT, qu_BI)
    precs, recs = topk(cateTrainTest, HammingRank1, topk=top_k)
    return precs, recs