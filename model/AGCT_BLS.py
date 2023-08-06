import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from AGCN import DAGCN
from Transformer import Transformer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from loguru import logger
from tensorboardX import SummaryWriter
import time
import random
from utils import get_all_result

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(42)


'''
重写pytorch dataset 类
相点是可以打乱的，相当于一个样本
'''

def normalize(default='MinMaxScaler'):
    if default == "StandardScaler":
        return StandardScaler()
    return MinMaxScaler(feature_range=(-1, 1))

class ChaoticDataset(Dataset):
    def __init__(self, win_size=100, slide_step=1,
                 predict_fature='x', predict_step=1,
                 train_length=3000, train_size=0.7,
                 data_path='../data/multi_bls_data',
                 type='lorenz', mode='train'):
        '''
        mode:train-->训练集的dataste,
        '''
        self.file_name = os.path.join(data_path, type, '%s_%s' % (predict_fature, predict_step))
        self.win_size = win_size
        self.slide_step = slide_step
        self.train_length = train_length
        self.train_size = train_size
        self.type = type
        self.mode = mode
        # self.scaler = normalize(default='MinMaxScaler')
        X = np.load(os.path.join(self.file_name, 'X.npy'), allow_pickle=True) # (3, 3935, 71)
        Y = np.load(os.path.join(self.file_name, 'Y.npy'), allow_pickle=True) # (3935, 1)

        # 标准化
        if self.type == 'lorenz' or self.type == 'rossler':
            self.features = ['x', 'y', 'z']
        if self.type == 'sea_clutter':
            self.features = ['feature_%s' % i for i in range(X.shape[0])]
        self.X = []
        for i in range(X.shape[0]):
            # 对每类序列进行标准化
            # feature = self.features[i]
            x_scaler = normalize(default='MinMaxScaler')
            x = x_scaler.fit_transform(X[i])
            self.X.append(x)
        self.X = np.array(self.X)
        self.y_scaler = normalize(default='MinMaxScaler')
        self.Y = self.y_scaler.fit_transform(Y)
        # 划分训练集、验证集、测试集
        assert self.X.shape[1] > self.train_length
        self.data = self.X[:,:self.train_length,:]        # [3,3000,71]
        self.label = self.Y[:self.train_length,:]         # [3000,1]
        self.test = self.X[:,self.train_length:,:]   # [3,935,71]
        self.test_label = self.Y[self.train_length:,:]    # [935,1]
        # train,valid
        self.train, self.val, self.train_label, self.val_label = [], [], [], []
        for i in range(self.data.shape[0]):
            data_i = self.data[i]
            label_i = self.label
            train_x, val_x, train_y, val_y = train_test_split(data_i, label_i, train_size=self.train_size,
                                                              shuffle=True,random_state=42)
            self.train.append(train_x)
            self.val.append(val_x)
            self.train_label = train_y
            self.val_label = val_y
        self.train = np.array(self.train)    # [3,2100,71]
        self.val = np.array(self.val)  # [3,900,71]

        print('-----train-----')
        print(f'train feature shape:{self.train.shape}')
        print(f'train label shape:{self.train_label.shape}')

        print('-----val-----')
        print(f'val feature shape:{self.val.shape}')
        print(f'val label shape:{self.val_label.shape}')

        print('-----test-----')
        print(f'test feature shape:{self.test.shape}')
        print(f'test label shape:{self.test_label.shape}')

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[1] - self.win_size) // self.slide_step + 1
        elif self.mode == "val":
            return (self.val.shape[1] - self.win_size) // self.slide_step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[1] - self.win_size) // self.slide_step + 1
        else:
            raise ValueError

    def __getitem__(self, item):
        index = item * self.slide_step # 每次向前跳步slide_step
        if self.mode == "train": # train, train_label
            return np.float32(self.train[:, index:index+self.win_size, :]), \
                self.train_label[index:index+self.win_size, :]
        elif self.mode == "val":
            return np.float32(self.val[:, index:index + self.win_size, :]), \
                self.val_label[index:index + self.win_size, :]
        elif self.mode == "test":
            return np.float32(self.test[:, index:index + self.win_size, :]), \
                self.test_label[index:index + self.win_size, :]
        else:
            raise ValueError


def get_data_loader(win_size=100, slide_step=1, predict_fature='x', predict_step=1,type='lorenz', mode='train', batch_size=100):
    dataset = ChaoticDataset(win_size, slide_step, predict_fature=predict_fature,
                             predict_step=predict_step, type=type, mode=mode)

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader


class AGCT_BLS(nn.Module):
    def __init__(self, num_time_steps, num_nodes, in_channels, hidden_dims, cheb_k, embed_dim,
                 out_channels=1, d_model=512, n_heads=8,
                 dropout=0.1, num_layers=1, output_attention=True
                 ):
        super(AGCT_BLS, self).__init__()
        self.spatial_model = DAGCN(num_time_steps=num_time_steps, num_nodes=num_nodes,
                                   in_dims=in_channels, out_dims=hidden_dims,
                                   cheb_k=cheb_k, embed_dim=embed_dim)
        self.temporal_model = Transformer(in_channels=hidden_dims, out_channels=out_channels,d_model=d_model,
                                          n_heads=n_heads, dropout=dropout, num_layers=num_layers,
                                          output_attention=output_attention)
        self.activation = F.relu

    def forward(self,x): # [B,N,L,C]
        # spatial
        x_s = self.spatial_model(x.permute(0,2,1,3))
        x_s = self.activation(x_s) # [B,L,N,hidden_dims]
        # temporal
        out, attns = self.temporal_model(x_s.permute(0,2,1,3)) # [B,L,1]
        return out, attns

def train(model, optimizer, criterion, scheduler, epoch, data_loader, sw, device):
# def train(model, optimizer, criterion, scheduler, epoch, data_loader, device):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch_index, batch_data in enumerate(data_loader):
        data, target = batch_data   # [B,N,L,C]-->[B,L,1]
        data = data.to(device).float()
        target = target.to(device).float()
        optimizer.zero_grad()
        output, attns = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        # 由于batch的存在有重复计算
        output1 = output[0,:,:] # [100,1]
        target1 = target[0,:,:] # [100,1
        for i in range(output.shape[0]):
            if i == 0:
                continue
            else:
                output1 = torch.cat((output1,output[i, -1,:].reshape(-1,1)),dim=0)
                target1 = torch.cat((target1,target[i, -1,:].reshape(-1,1)),dim=0)
        # 每个bach_index的损失
        loss1 = criterion(output1, target1)

        total_loss += loss1.item()
        log_interval = len(data_loader)/5  # 4
        if batch_index % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(
                epoch, batch_index, len(data_loader), scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss))  # , math.exp(cur_loss)
            total_loss = 0
            start_time = time.time()
    sw.add_scalar('training_loss', total_loss/len(data_loader), epoch)
    return total_loss/len(data_loader)
def evaluate(eval_model, data_loader, device):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for batch_index, batch_data in enumerate(data_loader):
            data, target = batch_data
            data = data.to(device)
            target = target.to(device)
            output, attns = eval_model(data)
            output1 = output[0, :, :]  # [100,1]
            target1 = target[0, :, :]  # [100,1
            for i in range(output.shape[0]):
                if i == 0:
                    continue
                else:
                    output1 = torch.cat((output1, output[i, -1, :].reshape(-1, 1)), dim=0)
                    target1 = torch.cat((target1, target[i, -1, :].reshape(-1, 1)), dim=0)
            # 每个bach_index的损失
            loss = criterion(output1, target1)
            total_loss += loss.item()
            if batch_index == 0:
                result = output[0, :, :].cpu()
                truth = target[0, :, :].cpu()
            else:
                for j in range(output.shape[0]):
                    result = torch.cat((result, output[j, -1, :].reshape(-1, 1).cpu()))
                    truth = torch.cat((truth, target[j, -1, :].reshape(-1, 1).cpu()))
    return total_loss / len(data_loader), result, truth


if __name__ == '__main__':
    train_loader = get_data_loader(win_size=1, slide_step=1,predict_fature='x', predict_step=1,
                                   type='lorenz', mode='train',
                                   batch_size=100)
    val_loader = get_data_loader(win_size=1, slide_step=1,predict_fature='x', predict_step=1,
                                   type='lorenz', mode='val',
                                   batch_size=100)
    test_loader = get_data_loader(win_size=1, slide_step=1,predict_fature='x', predict_step=1,
                                  type='lorenz', mode='test',
                                  batch_size=100)
    batch_size = 100  # batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用gpu
    criterion = nn.MSELoss()
    sw = SummaryWriter(logdir='../output/lorenz', flush_secs=5)
    lr = 0.0001
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    epochs = 100  # The number of epochs
    best_model = None
    model = AGCT_BLS(num_time_steps=1, num_nodes=3, in_channels=71, hidden_dims=512, cheb_k=2, embed_dim=4,
                     out_channels=1, d_model=512, n_heads=8,
                     dropout=0, num_layers=1, output_attention=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.96)
    # input, target = get_batch(train_seq, train_label, 0, batch_size)
    best_val_loss = 1000
    best_test_loss = 1000
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(model, optimizer, criterion, scheduler, epoch, train_loader, sw, device)
        # train_loss = train(model, optimizer, criterion, scheduler, epoch, train_loader, device)
        valid_loss, valid_output, valid_target = evaluate(model, val_loader, device)
        test_loss, test_output, test_target = evaluate(model, test_loader, device)
        sw.add_scalar('val_loss', valid_loss, epoch)
        sw.add_scalar('test_loss', test_loss, epoch)
        if valid_loss<best_val_loss:
            best_val_loss = valid_loss
            best_val_output = valid_output
            best_val_target = valid_target

        if test_loss<best_test_loss:
            best_test_loss = test_loss
            best_test_output = test_output
            best_test_target = test_target


        print('-' * 89)
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:.10f} | train loss {:.10f}| test loss {:.10f} '.format(
                epoch, (time.time() - epoch_start_time),
                valid_loss, train_loss, test_loss))
        scheduler.step()
    print('-------val-----')
    get_all_result(best_val_output, best_val_target)
    print('----test-----')
    get_all_result(best_test_output, best_test_target)
    plt.plot(best_test_output)
    plt.plot(best_test_target)





















