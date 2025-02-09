import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from evaluate import calculate_top_map
from load_dataset import load_dataset
from metric import ContrastiveLoss
from model_common_prompt import FuseTransEncoder, ImageMlp, TextMlp
from os import path as osp
from utils import load_checkpoints, save_checkpoints
from torch.optim import lr_scheduler
import time
import hdf5storage
from utils import cal_topK

# from b_reg import rand_unit_rect, gene_noise


class Solver(object):
    def __init__(self, config):
        self.config = config
        self.batch_size = 128
        self.total_epoch = config.epoch
        self.dataset = config.dataset
        self.model_dir = "./checkpoints"

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device(config.device if USE_CUDA else "cpu")

        self.task = config.task
        self.feat_lens = 512
        self.nbits = config.hash_lens
        # if self.config.dataset == 'nus-wide':
        #     num_layers, self.token_size, nhead = 2, 1024, 4
        # else:
        #     num_layers, self.token_size, nhead = 1, 1024, 2

        num_layers, self.token_size, nhead = 1, 1024, 2

        self.FuseTrans = FuseTransEncoder(num_layers, self.token_size, nhead).to(self.device)
        self.ImageMlp = ImageMlp(self.feat_lens, self.nbits).to(self.device)
        self.TextMlp = TextMlp(self.feat_lens, self.nbits).to(self.device)

        paramsFuse_to_update = list(self.FuseTrans.parameters())
        paramsImage = list(self.ImageMlp.parameters())
        paramsText = list(self.TextMlp.parameters())

        total_param = sum([param.nelement() for param in paramsFuse_to_update]) + sum(
            [param.nelement() for param in paramsImage]) + sum([param.nelement() for param in paramsText])
        print("total_param:", total_param)
        self.optimizer_FuseTrans = optim.Adam(paramsFuse_to_update, lr=1e-5, betas=(0.5, 0.999))
        self.optimizer_ImageMlp = optim.Adam(paramsImage, lr=1e-4, betas=(0.5, 0.999))
        self.optimizer_TextMlp = optim.Adam(paramsText, lr=1e-4, betas=(0.5, 0.999))

        if self.dataset == "mirflickr" or self.dataset == "nus-wide":
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[30, 80], gamma=1.2)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[30, 80], gamma=1.2)
        elif self.dataset == "mscoco":
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[200], gamma=0.6)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[200], gamma=0.6)

        data_loader = load_dataset(self.dataset, self.batch_size)
        self.train_loader = data_loader['train']
        self.query_loader = data_loader['query']
        self.retrieval_loader = data_loader['retrieval']

        self.ContrastiveLoss = ContrastiveLoss(batch_size=self.batch_size, device=self.device)

        self.maxfunc = torch.nn.ReLU()

        self.best_it = self.best_ti = 0

    def train(self):

        if self.task == 1:  # train hash
            print("Training Hash Fuction...")
            I2T_MAP = []
            T2I_MAP = []
            start_time = time.time()
            for epoch in range(self.total_epoch):
                print("epoch:", epoch + 1)
                train_loss = self.trainhash()
                print(train_loss)
                if (epoch + 1) % 10 == 0:
                    print("Testing...")
                    img2text, text2img = self.evaluate()
                    I2T_MAP.append(img2text)
                    T2I_MAP.append(text2img)
                    print('I2T:', img2text, ', T2I:', text2img)

                    # -----------------------------------------------------------
                    if (self.best_it + self.best_ti) < (img2text + text2img):
                        self.best_it, self.best_ti = img2text, text2img

                        print(f'Best MAP of I->T: {self.best_it}')
                        print(f'Best MAP of T->I: {self.best_ti}')
                        save_checkpoints(self)
                    # -----------------------------------------------------------
            print(f'Best MAP of I->T: %.3f, Best mAP of T->I: %.3f' % (self.best_it, self.best_ti))

            print(I2T_MAP, T2I_MAP)

            time_elapsed = time.time() - start_time
            print(f'Total Train Time: {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')

            return (img2text + text2img) / 2., img2text, text2img


    def evaluate(self):
        self.FuseTrans.eval()
        self.ImageMlp.eval()
        self.TextMlp.eval()
        qu_BI, qu_BT, qu_L = [], [], []
        re_BI, re_BT, re_L = [], [], []
        if self.config.EVAL == False:
            with torch.no_grad():
                for _, (data_I, data_T, data_L, _) in enumerate(self.query_loader):
                    data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                    temp_tokens = torch.concat((data_I, data_T), dim=1)
                    # temp_tokens = temp_tokens.unsqueeze(0)
                    # img_query, txt_query = self.FuseTrans(temp_tokens)
                    common_query, query_scalar = self.FuseTrans(temp_tokens)
                    img_query = common_query + query_scalar * data_I
                    txt_query = common_query + query_scalar * data_T
                    if self.task == 1 or self.task == 3:
                        img_query = self.ImageMlp(img_query)
                        txt_query = self.TextMlp(txt_query)
                    img_query, txt_query = img_query.cpu().numpy(), txt_query.cpu().numpy()
                    qu_BI.extend(img_query)
                    qu_BT.extend(txt_query)
                    qu_L.extend(data_L.cpu().numpy())

                for _, (data_I, data_T, data_L, _) in enumerate(self.retrieval_loader):
                    data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                    temp_tokens = torch.concat((data_I, data_T), dim=1)
                    # temp_tokens = temp_tokens.unsqueeze(0)
                    # img_retrieval, txt_retrieval = self.FuseTrans(temp_tokens)
                    common_retrieval, retrieval_scalar = self.FuseTrans(temp_tokens)
                    img_retrieval = common_retrieval + retrieval_scalar * data_I
                    txt_retrieval = common_retrieval + retrieval_scalar * data_T
                    if self.task == 1 or self.task == 3:
                        img_retrieval = self.ImageMlp(img_retrieval)
                        txt_retrieval = self.TextMlp(txt_retrieval)
                    img_retrieval, txt_retrieval = img_retrieval.cpu().numpy(), txt_retrieval.cpu().numpy()
                    re_BI.extend(img_retrieval)
                    re_BT.extend(txt_retrieval)
                    re_L.extend(data_L.cpu().numpy())

            re_BI = np.array(re_BI)
            re_BT = np.array(re_BT)
            re_L = np.array(re_L)

            qu_BI = np.array(qu_BI)
            qu_BT = np.array(qu_BT)
            qu_L = np.array(qu_L)

            if self.task == 1 or self.task == 3:  # hashing
                qu_BI = torch.sign(torch.tensor(qu_BI)).cpu().numpy()
                qu_BT = torch.sign(torch.tensor(qu_BT)).cpu().numpy()
                re_BT = torch.sign(torch.tensor(re_BT)).cpu().numpy()
                re_BI = torch.sign(torch.tensor(re_BI)).cpu().numpy()
            elif self.task == 0 or self.task == 2:  # real value
                qu_BI = torch.tensor(qu_BI).cpu().numpy()
                qu_BT = torch.tensor(qu_BT).cpu().numpy()
                re_BT = torch.tensor(re_BT).cpu().numpy()
                re_BI = torch.tensor(re_BI).cpu().numpy()

            MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
            MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        else:
            # -----------------------------------------------------------------------------------------
            top_k_p_I2T = []
            top_k_r_I2T = []

            top_k_p_T2I = []
            top_k_r_T2I = []
            # -----------------------------------------------------------------------------------------
            with torch.no_grad():
                for _, (data_I, data_T, data_L, _) in enumerate(self.query_loader):
                    data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                    temp_tokens = torch.concat((data_I, data_T), dim=1)
                    # temp_tokens = temp_tokens.unsqueeze(0)
                    # img_query, txt_query = self.FuseTrans(temp_tokens)
                    common_query, query_scalar = self.FuseTrans(temp_tokens)
                    img_query = common_query + query_scalar * data_I
                    txt_query = common_query + query_scalar * data_T
                    if self.task == 1 or self.task == 3:
                        img_query = self.ImageMlp(img_query)
                        txt_query = self.TextMlp(txt_query)
                    img_query, txt_query = img_query.cpu().numpy(), txt_query.cpu().numpy()
                    qu_BI.extend(img_query)
                    qu_BT.extend(txt_query)
                    qu_L.extend(data_L.cpu().numpy())

                for _, (data_I, data_T, data_L, _) in enumerate(self.retrieval_loader):
                    data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                    temp_tokens = torch.concat((data_I, data_T), dim=1)
                    # temp_tokens = temp_tokens.unsqueeze(0)
                    # img_retrieval, txt_retrieval = self.FuseTrans(temp_tokens)
                    common_retrieval, retrieval_scalar = self.FuseTrans(temp_tokens)
                    img_retrieval = common_retrieval + retrieval_scalar * data_I
                    txt_retrieval = common_retrieval + retrieval_scalar * data_T
                    if self.task == 1 or self.task == 3:
                        img_retrieval = self.ImageMlp(img_retrieval)
                        txt_retrieval = self.TextMlp(txt_retrieval)
                    img_retrieval, txt_retrieval = img_retrieval.cpu().numpy(), txt_retrieval.cpu().numpy()
                    re_BI.extend(img_retrieval)
                    re_BT.extend(txt_retrieval)
                    re_L.extend(data_L.cpu().numpy())

            re_BI = np.array(re_BI)
            re_BT = np.array(re_BT)
            re_L = np.array(re_L)

            qu_BI = np.array(qu_BI)
            qu_BT = np.array(qu_BT)
            qu_L = np.array(qu_L)

            if self.task == 1 or self.task == 3:  # hashing
                qu_BI = torch.sign(torch.tensor(qu_BI)).cpu().numpy()
                qu_BT = torch.sign(torch.tensor(qu_BT)).cpu().numpy()
                re_BT = torch.sign(torch.tensor(re_BT)).cpu().numpy()
                re_BI = torch.sign(torch.tensor(re_BI)).cpu().numpy()

            elif self.task == 0 or self.task == 2:  # real value
                qu_BI = torch.tensor(qu_BI).cpu().numpy()
                qu_BT = torch.tensor(qu_BT).cpu().numpy()
                re_BT = torch.tensor(re_BT).cpu().numpy()
                re_BI = torch.tensor(re_BI).cpu().numpy()

            MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
            MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

            # ----------------------------------------------------------------------------------------
            # K = [1, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000,
            #      4250, 4500, 4750, 5000]

            K = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
                 950, 1000]

            for idx, i in enumerate(K):
                p_I2T, r_I2T = cal_topK(qu_BI=qu_BI, re_BT=re_BT, test_label=qu_L, train_label=re_L, top_k=i)
                p_T2I, r_T2I = cal_topK(qu_BI=qu_BT, re_BT=re_BI, test_label=qu_L, train_label=re_L, top_k=i)
                top_k_p_I2T.append(p_I2T)
                top_k_r_I2T.append(r_I2T)
                top_k_p_T2I.append(p_T2I)
                top_k_r_T2I.append(r_T2I)

            data_map = {'MAP_I2T': MAP_I2T, 'MAP_T2I': MAP_T2I}

            data = {'R_I2T': top_k_r_I2T, 'P_I2T': top_k_p_I2T, 'R_T2I': top_k_r_T2I, 'P_T2I': top_k_p_T2I}

            Method = 'UCMFH'
            hdf5storage.savemat(Method + '_' + self.config.dataset + str(self.config.hash_lens) + '1000' + '.mat', data)
            hdf5storage.savemat(
                'MAP' + Method + '_' + self.config.dataset + str(self.config.hash_lens) + '1000' + '.mat', data_map)
            # ----------------------------------------------------------------------------------------

        return MAP_I2T, MAP_T2I

    def trainhash(self):
        self.FuseTrans.train()
        self.ImageMlp.train()
        self.TextMlp.train()
        running_loss = 0.0
        for idx, (img, txt, labels, _) in enumerate(self.train_loader):
            img, txt = img.to(self.device), txt.to(self.device)
            temp_tokens = torch.concat((img, txt), dim=1)
            temp_tokens = temp_tokens.unsqueeze(0)
            # img_embedding, text_embedding = self.FuseTrans(temp_tokens)
            common_embedding, scalar = self.FuseTrans(temp_tokens)

            img_embedding = common_embedding + scalar * img
            text_embedding = common_embedding + scalar * txt

            img_embedding = self.ImageMlp(img_embedding)
            text_embedding = self.TextMlp(text_embedding)
            loss_contra = self.ContrastiveLoss(img_embedding, text_embedding)

            loss = loss_contra
            self.optimizer_FuseTrans.zero_grad()
            self.optimizer_ImageMlp.zero_grad()
            self.optimizer_TextMlp.zero_grad()
            loss.backward()
            self.optimizer_FuseTrans.step()
            self.optimizer_ImageMlp.step()
            self.optimizer_TextMlp.step()
            running_loss += loss.item()

            self.ImageMlp_scheduler.step()
            self.TextMlp_scheduler.step()
        return running_loss
