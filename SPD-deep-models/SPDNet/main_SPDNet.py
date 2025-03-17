import sys
sys.path.append('/home/zju/Python_Scripts/Riemannian_RNN/EcoGLibrary/SEED')

import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os, argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import covariance
from sklearn.model_selection import KFold
from scipy import signal
import scipy.io as sio

from Model.Model_SPDNet.spd import MySPDNet
from SPDNet.train_SPDNet import train_op, test_op
from Model.utils import BaseDataset_static, calculate_metrics


def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Sleep stage classification')
    parser.add_argument('-g', '--gpu', default='1', help='GPU number')
    parser.add_argument('--subjects', default=45, type=int, help='Number of subjects')
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs')
    parser.add_argument('--latents', default=31, type=int, help='Latents dimension')
    parser.add_argument('--units', default=16, type=int)
    parser.add_argument('-T', default=5, type=int)
    parser.add_argument('-classes', default=2, type=int)
    parser.add_argument('-channel', default=1, type=int)
    parser.add_argument('-bi', default=False)

    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--seed', default=100, help='Random seed')
    parser.add_argument('--model', default='SPDNet')

    parser.add_argument('--path', default='/data/lalalagegrx/RiemannianRNN/EcoGLibrary/SEED/', help='Directory')
    parser.add_argument('--batch', default=16, type=int, help='Batch size')
    parser.add_argument('--fold', default=15, type=int, help='Fold number')
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda:2' if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    idx_ch = [15, 23, 24, 25, 31, 32, 33, 34, 40, 41, 42, 50]
    acc_l, sens_l, spec_l, mf1_l, kappa_l = [], [], [], [], []
    for sub in range(1, args.subjects + 1):
        print('Subject ', sub)
        path = args.path + 'Data_S{}.mat'
        Data = sio.loadmat(path.format(sub))
        X1 = Data['Data'][:, idx_ch, :].astype('float32')
        Y1 = Data['Label'].squeeze().astype('float32')
        twoclass_indices = np.where((Y1 == -1) | (Y1 == 1))
        X1 = X1[twoclass_indices]
        Y1 = Y1[Y1 != 0]
        Y1[Y1 == -1] = 0 
        args.latents = len(idx_ch)
        args.units = math.ceil(args.latents / 2)
        #for i in range(args.latents):
        #    X1[:, :, i, i] += 0.01

        # Dataset
        dataset = BaseDataset_static(X1, Y1, args)
        kfold = KFold(n_splits = args.fold, shuffle=False)
        preds, targets = [], []
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            start_time = time.time()
            print(f'FOLD {fold}')
            print('--------------------------------')
        
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            trDL = DataLoader(dataset, batch_size=args.batch, sampler=train_subsampler)
            valDL = DataLoader(dataset, batch_size=args.batch, sampler=test_subsampler)

            cache = {'Model': args.model, 'Fold': args.fold, "Acc":0, 'F1':0, 'Kappa':0, 'Sens':0, 'Spec':0, 'val_loss':10, 'Epoch':0}

            if args.model == "SPDNet":
                model = MySPDNet(args.classes, args.latents).to(device)

            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

            # train
            for epoch in range(args.epochs):
                print("Epoch {} Start.".format(epoch))
                _ = train_op(model, device, trDL, optimizer, criterion)
            
            # valid
            val_loss, ys, h_ys = test_op(model, device, valDL, criterion, type='Valid')
            ys = np.concatenate(ys)
            h_ys = np.concatenate(h_ys)
            for i in range(len(ys)):
                y = np.log(h_ys[i])
                preds.append(np.argmax(y))
                targets.append(ys[i])

            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")
            exit()

        acc, sens, spec, mf1, kappa = calculate_metrics(np.array(targets), np.array(preds))
        print("Test||Acc:{:.4f}, F1:{:.4f}, Kappa:{:.4f}, Sens:{:.4f}, Spec:{:.4f}".format(acc, mf1, kappa, sens, spec))
        # if acc > cache['Acc'] or (acc == cache['Acc'] and mf1 > cache['F1']):
        #     # acc, sens, spec, mf1, kappa = calculate_metrics(targets, preds)
        #     cache['val_loss'] = val_loss
        #     cache['Acc'] = round(acc, 4)
        #     cache['F1'] = round(mf1, 4)
        #     cache['Kappa'] = round(kappa, 4)
        #     cache["Sens"] = round(sens, 4)
        #     cache['Spec'] = round(spec, 4)
        #     cache['Epoch'] = epoch
        #     state = {
        #         'model': model.state_dict(),
        #         'pred': h_ys
        #     }

        # print("-"*5, "Fold:{}, Epoch:{} || Acc:{:.4f}, F1:{:.4f}, Kappa:{:.4f}, Sens:{:.4f}, Spec:{:.4f}".format(cache['Fold'],
        #                                                                                     cache['Epoch'],
        #                                                                                     cache['Acc'],
        #                                                                                     cache['F1'],
        #                                                                                     cache['Kappa'],
        #                                                                                     cache['Sens'],
        #                                                                                     cache['Spec']))
        acc_l.append(acc)
        mf1_l.append(mf1)

    acc_mean = sum(acc_l) / args.subjects
    mf1_mean = sum(mf1_l) / args.subjects
    print(sum(acc_l) / args.subjects)
    print(sum(mf1_l) / args.subjects)
