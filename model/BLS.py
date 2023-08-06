import numpy as np
from sklearn import preprocessing
from numpy import random
import time
from utils import get_all_result
import pandas as pd
import matplotlib.pyplot as plt
import d2l.torch as d2l

def tansig(x):
    return (2/(1+np.exp(-2*x)))-1

# 稀疏化编码mapping features
def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z

def sparse_bls(A,b):
    lam = 0.001
    itrs = 50
    AA = np.dot(A.T,A)
    m = A.shape[1]
    n = b.shape[1]
    wk = np.zeros([m,n],dtype = 'double')
    ok = np.zeros([m,n],dtype = 'double')
    uk = np.zeros([m,n],dtype = 'double')
    L1 = np.mat(AA + np.eye(m)).I
    L2 = np.dot(np.dot(L1,A.T),b)
    for i in range(itrs):
        tempc = ok - uk
        ck =  L2 + np.dot(L1,tempc)
        ok = shrinkage(ck + uk, lam)
        uk += ck - ok
        wk = ok
    return wk

def pinv(A,reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)


class BLS:
    def __init__(self, map_fea_num=6, map_num=5, enh_fea_num=41, enh_num=1, c=2 ** -30):
        self.map_fea_num = map_fea_num
        self.map_num = map_num
        self.enh_fea_num = enh_fea_num
        self.enh_num = enh_num
        self.c = c

    def generator_mapping_features(self, input_channels):
        self.map_weights = []
        for i in range(self.map_num):
            random.seed(i) # [0,1]-->random.random
            map_fea_weight = 2 * random.randn(input_channels + 1, self.map_fea_num) - 1  # [5+1,6]
            self.map_weights.append(map_fea_weight)

    def generator_enhance_features(self):
        self.enhance_weights = []
        for i in range(self.enh_num):
            random.seed(i)
            enh_fea_weight = 2 * random.randn(self.map_num * self.map_fea_num + 1, self.enh_fea_num) - 1;  # [5*6+1,41]
            self.enhance_weights.append(enh_fea_weight)

    def sparse_autoencoder_weights(self, x):  # x-->[3000,5]
        H1 = np.hstack([x, 0.1 * np.ones([x.shape[0], 1])]) # [3000,6]
        self.map_features = np.zeros([x.shape[0], self.map_num * self.map_fea_num])
        self.wf_sparse = list()
        self.distOfMaxAndMin = np.zeros(self.map_num)
        self.meanOfEachWindow = np.zeros(self.map_num)
        for i in range(self.map_num):
            map_fea_weight = self.map_weights[i]  # [6,6]
            A1 = H1.dot(map_fea_weight)
            scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
            A1 = scaler1.transform(A1)
            map_sparse_fea_weight = sparse_bls(A1, H1).T
            self.wf_sparse.append(map_sparse_fea_weight)

            # 重新得到映射特征
            T1 = H1.dot(map_sparse_fea_weight)
            self.meanOfEachWindow[i] = T1.mean()
            self.distOfMaxAndMin[i] = T1.max() - T1.min()
            T1 = (T1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i]  # 标准化T1-->[3000,6]
            # print(T1.shape)
            self.map_features[:, self.map_fea_num * i:self.map_fea_num * (i + 1)] = T1

    def concat_mapping_enhance_features(self):
        self.combine_features = self.map_features
        for i in range(self.enh_num):
            enh_weight = self.enhance_weights[i]
            H2 = np.hstack([self.map_features, 0.1 * np.ones([self.map_features.shape[0], 1])])  # [3000,31]
            T2 = H2.dot(enh_weight)  # [3000,41]
            T2 = tansig(T2)
            self.combine_features = np.hstack([self.combine_features, T2])  # [3000,71]

    def generate_features(self, x, is_train=True):
        if is_train:
            _, input_channels = x.shape
            self.generator_mapping_features(input_channels)
            self.generator_enhance_features()
            self.sparse_autoencoder_weights(x)
            self.concat_mapping_enhance_features()
            # print(f'combine features shape:{self.combine_features.shape}')
            return self.combine_features
        else:
            HH1 = np.hstack([x, 0.1 * np.ones([x.shape[0], 1])])
            yy1 = np.zeros([x.shape[0], self.map_num * self.map_fea_num])
            for i in range(self.map_num):
                map_sparse_fea_weight = self.wf_sparse[i]  # 这里至关重要他是从训练集计算来的
                # 计算测试集的稀疏映射特征
                TT1 = HH1.dot(map_sparse_fea_weight)
                TT1 = (TT1 - self.meanOfEachWindow[i]) / self.distOfMaxAndMin[i] # min,max也是训练集得来的
                yy1[:, self.map_fea_num * i:self.map_fea_num * (i + 1)] = TT1
            combine_features = yy1
            for i in range(self.enh_num):
                enh_weight = self.enhance_weights[i]
                HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0], 1])])
                TT2 = tansig(HH2.dot(enh_weight))
                combine_features = np.hstack([combine_features, TT2])  # [3000,71]
            return combine_features

    def fit(self, train_x, train_y):
        combine_features = self.generate_features(train_x)  # [3000,71]
        self.weight_last = pinv(combine_features, self.c).dot(train_y)  # [71,1]
        train_out = combine_features.dot(self.weight_last)
        print(f'train out shape:{train_out.shape}')
        print('train performance:')
        get_all_result(train_out, train_y)

    def predict(self, test_x,test_y):
        test_combine_features = self.generate_features(test_x, is_train=False)
        test_out = test_combine_features.dot(self.weight_last)
        print(f'test out shape:{test_out.shape}')
        print('test performance:')
        get_all_result(test_out, test_y)
        return test_out


if __name__ == "__main__":
    d2l.use_svg_display()
    import sys
    sys.path.append('E:\学术生涯\研究生生涯\个人项目论文\多维混沌时间序列预测\AGCT-BLS\data')
    data = np.load('../data/standard_data/sea_clutter_rec_dict.npy', allow_pickle=True).tolist()
    # data = np.load('../data/standard_data/sea_clutter_rec_dict.npy', allow_pickle=True).tolist()
    # data = data['x']
    data = data['feature_0']
    print(data.shape)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    train_x = data[:3000]
    train_y = data[1:3001,-1].reshape(-1,1)
    test_x = data[3000:-1,:]
    test_y = data[3001:,-1].reshape(-1,1)
    bls = BLS(map_fea_num=6, map_num=5, enh_fea_num=41, enh_num=1)
    # combine_features = bls.generate_features(x)
    bls.fit(train_x, train_y)
    test_out = bls.predict(test_x,test_y)
    plt.plot(test_out[:10])
    plt.plot(test_y[:10])
    plt.show()


