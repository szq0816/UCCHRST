import torch
import numpy as np
import os
import argparse
from ICMR_common_prompt import Solver
import time

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    seeds = 2023
    torch.manual_seed(seeds)
    torch.cuda.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    np.random.seed(seeds)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nus-wide', help='Dataset name: mirflickr, mscoco, nus-wide')
    parser.add_argument('--epoch', type=int, default=70, help='default:50 epochs')
    parser.add_argument('--hash_lens', type=int, default=16)
    parser.add_argument('--device', type=str, default="cuda:0", help='cuda device')
    parser.add_argument('--task', type=int, default=1,
                        help="0 is train real; 1 is train hash; 2 is test real; 3 is test hash")
    parser.add_argument('--EVAL', type=bool, default=False, help='train or test')
    config = parser.parse_args()
    # ----------------------------------------------------------
    config.MU = 1.5
    config.ETA = 0.4
    config.BATCH_SIZE = 128

    if config.dataset == 'mirflickr':
        config.BETA = 0.9
        config.LAMBDA1 = 0.1
        config.LAMBDA2 = 0.1
        config.contra = 1

    if config.dataset == 'nus-wide':
        config.BETA = 0.6
        config.LAMBDA1 = 0.1
        config.LAMBDA2 = 0.1
        config.contra = 1

    if config.dataset == 'mscoco':
        config.BETA = 0.6
        config.LAMBDA1 = 0.1
        config.LAMBDA2 = 0.1
        config.contra = 1
    # -----------------------------------------------------------

    config = parser.parse_args()
    for i in [16, 32, 64, 128]:
    # for i in [16]:
        config.hash_lens = i

        start_time = time.time()
        Final_mean_MAP = {}
        Final_std_MAP = {}
        total_map = []
        map_i2t = []
        map_t2i = []
        task = str(config.hash_lens) + " bits" if (config.task in [1, 3]) else "real value"
        print('=============== {}--{}--Total epochs:{} ==============='.format(config.dataset, task, config.epoch))
        # for state in [1,2,3,4,5]:
        for state in [1]:
            print('...Training is beginning...', state)
            solver = Solver(config)
            map, i2t, t2i = solver.train()
            total_map.append(map)
            map_i2t.append(i2t)
            map_t2i.append(t2i)

        mean_map = round(np.mean(total_map), 4)
        std_map = round(np.std(total_map), 4)
        i2t_map = round(np.mean(map_i2t), 4)
        t2i_map = round(np.mean(map_t2i), 4)
        print("mean_map:{0}, std_map:{1} , i2t:{2}, t2t:{3} .".format(mean_map, std_map, i2t_map, t2i_map))
        time_elapsed = time.time() - start_time
        print(f'Total Time: {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
        with open('result.txt', 'a+', encoding='utf-8') as f:
            f.write("[{0}-{1}-{2} epochs]".format(config.dataset, config.hash_lens, config.epoch))
            f.write("mean:{0}, std:{1},i2t:{2}, t2t:{3}.\n".format(mean_map, std_map, i2t_map, t2i_map))
