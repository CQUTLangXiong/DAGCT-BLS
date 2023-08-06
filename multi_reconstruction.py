import numpy as np
import time
import pandas as pd
from numba import jit


def get_data_path(type='lorenz'):
    if type == 'lorenz':
        data_path = 'data/original_data/lorenz.csv'
        data = pd.read_csv(data_path)
    elif type == 'rossler':
        data_path = 'data/original_data/rossler.csv'
        data = pd.read_csv(data_path)
    elif type == 'sea_clutter':
        data_path = 'data/original_data/sea_clutter.xlsx'
        data = pd.read_excel(data_path, header=None)
        data.columns = ['feature_' + str(column) for column in data.columns]

    else:
        raise
    return data

def reconstruction(data, m, tau):
    """
    该函数用来重构相空间
    m:嵌入维数
    tau：时间延迟
    return:rec_data-->(m,n)
    """
    n = len(data)
    M = n - (m-1) * tau
    rec_data = np.zeros([m, M])
    # Data = np.zeros([m, M])
    for j in range(M):
        for i in range(m):
            rec_data[i, j] = data[i*tau+j]
    return rec_data

def multi_reconstruction(data_type='lorenz', data_range=(3000,7000), save=True):
    data = get_data_path(data_type)
    data = data.iloc[data_range[0]:data_range[1], :]
    # 初始化字典存取对应的特征的重构数据
    rec_dict = {}
    # feature
    if data_type=='lorenz' or data_type=='rossler':
        features = ['x', 'y', 'z']
    else:
        features = ['feature_%s' % i for i in range(data.shape[1])]
    # 读取对应数据的嵌入维度和延迟tau
    data_tau_m = pd.read_excel('data/%s_tau_m.xlsx' % data_type)
    for feature in features:
        # 读取对应的tau,m
        data_feature = data[feature].values
        tau = data_tau_m[data_tau_m['feature'] == feature]['tau'].tolist()[0]
        m = data_tau_m[data_tau_m['feature'] == feature]['m'].tolist()[0]
        rec_data = reconstruction(data_feature, m, tau)
        rec_dict[feature] = rec_data.T
    if save:
        np.save('data/reconstruction_data/%s_rec_dict.npy' % data_type, rec_dict)
    return rec_dict

# 取最小的样本个数,获取标准数据
def get_final_result(type='lorenz', save=True):
    # data_rec_dict = np.load('data/reconstruction_data/%s_rec_dict.npy' % type, allow_pickle=True).tolist()
    data_rec_dict =  multi_reconstruction(data_type='lorenz', data_range=(3000,7000))
    # 寻找最小的样本量
    min_sample_num = np.inf
    for key, value in data_rec_dict.items():
        sample_num = value.shape[0]
        min_sample_num = min(min_sample_num, sample_num)
    # 根据min_sample_num取出对应的数据
    print(f'all features min sample number:{min_sample_num}')
    for key, value in data_rec_dict.items():
        value = value[:min_sample_num,:]
        data_rec_dict[key] = value
    if save:
        np.save('data/standard_data/%s_rec_dict.npy' % type, data_rec_dict)
    return data_rec_dict


if __name__ == "__main__":
    data_rec_dict = get_final_result(type='lorenz', save=True)
    print(f"lorenz x shape:{data_rec_dict['x'].shape}")
    print(f"lorenz x shape:{data_rec_dict['y'].shape}")
    print(f"lorenz x shape:{data_rec_dict['z'].shape}")





