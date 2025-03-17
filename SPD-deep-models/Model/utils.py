import torch
from torch.utils.data import Dataset
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
import math
import numpy as np
import csv


class BaseDataset(Dataset):
    def __init__(self, raw, target, args):
        [trials, channels, T] = raw.shape
        window_time = 1
        num_window = T // 200
        windowed_data = raw.reshape(trials, channels, num_window, 200).transpose(0, 2, 1, 3)
        X = np.zeros((trials, num_window, channels, channels))
        for i in range(trials):
            for j in range(num_window):
                window = windowed_data[i, j, :, :]
                X[i, j] = np.cov(window)
        

        self.X = X.reshape((trials*(num_window//args.T), args.T, channels, channels))
        self.target = np.repeat(target, num_window // args.T)
        self.args = args

    def __getitem__(self, index):
        return self.X[index], self.target[index]

    def __len__(self):
        return len(self.X)

class BaseDataset_static(Dataset):
    def __init__(self, raw, target, args):
        [trials, channels, T] = raw.shape
        window_time = 1
        num_window = T // (args.T * 200)
        windowed_data = raw.reshape(trials, channels, num_window, args.T * 200).transpose(0, 2, 1, 3)
        X = np.zeros((trials, num_window, channels, channels))
        for i in range(trials):
            for j in range(num_window):
                window = windowed_data[i, j, :, :]
                X[i, j] = np.cov(window)
        
        self.X = X.reshape((trials*num_window, channels, channels))
        self.target = np.repeat(target, num_window)
        self.args = args

    def __getitem__(self, index):
        return self.X[index], self.target[index]

    def __len__(self):
        return len(self.X)

class BaseDataset_RSSM(Dataset):
    def __init__(self, raw, target, samples_cumsum, args):
        channels = raw.shape[0]
        trials = 10
        window = 5
        X = []
        y = []
        for i in range(trials):
            trial_data = raw[:, :, samples_cumsum[i]: samples_cumsum[i+1]].transpose(2, 0, 1)
            samples = trial_data.shape[0]
            trial_data = trial_data[:(samples // window)*window]
            trial_data = trial_data.reshape((samples // window, window, channels, channels))
            trial_label = target[i] * np.ones(trial_data.shape[0])
            X.append(trial_data)
            y.append(trial_label)
        self.X = np.concatenate(X)
        self.target = np.concatenate(y)
        self.args = args

    def __getitem__(self, index):
        return self.X[index], self.target[index]

    def __len__(self):
        return len(self.X)
    
class BaseDataset_RSSM_static(Dataset):
    def __init__(self, raw, target, samples_cumsum, args):
        channels = raw.shape[0]
        trials = 10
        window = 5
        X = []
        y = []
        for i in range(trials):
            trial_data = raw[:, :, samples_cumsum[i]: samples_cumsum[i+1]].transpose(2, 0, 1)
            trial_label = target[i] * np.ones(trial_data.shape[0])
            X.append(trial_data)
            y.append(trial_label)
        self.X = np.concatenate(X)
        self.target = np.concatenate(y)
        self.args = args

    def __getitem__(self, index):
        return self.X[index], self.target[index]

    def __len__(self):
        return len(self.X)

def save_csv(filename, cache):
    with open("./{}_results.csv".format(filename), 'a') as f:
        w = csv.writer(f)
        if f.tell() == 0: w.writerow(cache.keys())
        w.writerow(cache.values())

def calculate_metrics(y, y_pred):
    #acc, sens, spec, mf1, kappa
    def average_sen_spec(y, y_pred):
        tn = multilabel_confusion_matrix(y, y_pred)[:, 0, 0]
        fn = multilabel_confusion_matrix(y, y_pred)[:, 1, 0]
        tp = multilabel_confusion_matrix(y, y_pred)[:, 1, 1]
        fp = multilabel_confusion_matrix(y, y_pred)[:, 0, 1]
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        return sens.mean(), spec.mean()
    acc = accuracy_score(y, y_pred)
    sens, spec = average_sen_spec(y, y_pred)
    mf1 = f1_score(y, y_pred, average='macro')
    kappa = cohen_kappa_score(y, y_pred)
    return acc, sens, spec, mf1, kappa
