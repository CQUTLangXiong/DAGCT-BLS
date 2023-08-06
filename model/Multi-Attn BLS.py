#导入基本块
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import d2l.torch as d2l
from utils import get_all_result
d2l.use_svg_display()

#环境设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(42)
np.random.seed(42)
calculate_loss_over_all_values = False
input_window = 100
output_window = 1
batch_size = 100  # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#使用gpu

criterion = nn.MSELoss()
lr = 0.0001
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
best_val_loss = float("inf")
epochs =100# The number of epochs
best_model = None
def get_bls_data(data_path='../data/multi_bls_data/', type='lorenz',
             predict_fature='x', predict_step=1, is_split=True):
    file_name = os.path.join(data_path, type, '%s_%s' % (predict_fature, predict_step))
    if not is_split:
        X = np.load(os.path.join(file_name, 'X.npy'), allow_pickle=True)
        Y = np.load(os.path.join(file_name, 'Y.npy'), allow_pickle=True)
        print(f'All bls X shape:{X.shape}')
        print(f'All y shape:{Y.shape}')
        return X, Y
    X_train_bls = np.load(os.path.join(file_name, 'X_train_bls.npy'), allow_pickle=True)
    X_test_bls = np.load(os.path.join(file_name, 'X_test_bls.npy'), allow_pickle=True)
    train_y = np.load(os.path.join(file_name, 'train_y.npy'), allow_pickle=True)
    test_y = np.load(os.path.join(file_name, 'test_y.npy'), allow_pickle=True)
    print(f'train bls X shape:{X_train_bls.shape}')
    print(f'test bls X shape:{X_test_bls.shape}')
    print(f'train Y shape:{train_y.shape}')
    print(f'test Y shape:{test_y.shape}')
    return X_train_bls, X_test_bls, train_y, test_y


def get_data(X_broad, y_broad):
    # 取前3000的数据进行训练和验证，后面的数据进行测试
    # 训练和验证
    # data = X_broad.iloc[:3000, :]
    # test_data = X_broad.iloc[3000:, :]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_broad = scaler.fit_transform(X_broad)
    data = X_broad[:3000, :]
    test_data = X_broad[3000:, :]
    #     # 标准化为-1到1
    # data = scaler.fit_transform(data)
    # test_data = scaler.fit_transform(test_data)
    # 训练和验证标签
    # label = y_broad.iloc[:3000, :]
    # label = scaler.fit_transform(label)
    # test_label = y_broad.iloc[3000:, :]
    y_broad = scaler.fit_transform(y_broad)
    label = y_broad[:3000, :]
    # label = scaler.fit_transform(label)
    test_label = y_broad[3000:, :]
    # test_label = scaler.fit_transform(test_label)
    # indices = np.arange(len(data))
    # 拆分训练集和验证集
    train_X, valid_X, train_y, valid_y= train_test_split(data, label, train_size=0.7,shuffle=True,
                                                                      random_state=42)

    # 将数据转化为Tensor
    # 训练
    train_seq = torch.from_numpy(np.array(train_X)).type(torch.FloatTensor)
    train_label = torch.from_numpy(np.array(train_y)).type(torch.FloatTensor)
    # 验证集
    valid_seq = torch.from_numpy(np.array(valid_X)).type(torch.FloatTensor)
    valid_label = torch.from_numpy(np.array(valid_y)).type(torch.FloatTensor)
    # 测试集
    test_seq = torch.from_numpy(np.array(test_data)).type(torch.FloatTensor)
    test_label = torch.from_numpy(np.array(test_label)).type(torch.FloatTensor)
    print('----------train--------')
    print(f'train X shape:{train_seq.shape}')
    print(f'train y shape:{train_label.shape}')

    print('----------val--------')
    print(f'val X shape:{valid_seq.shape}')
    print(f'val y shape:{valid_label.shape}')

    print('----------test--------')
    print(f'test X shape:{test_seq.shape}')
    print(f'test y shape:{test_label.shape}')

    return train_seq.to(device), train_label.to(device), \
        valid_seq.to(device), valid_label.to(device), \
        test_seq.to(device), test_label.to(device)


def get_batch(seq, label, i, batch_size):
    seq_len = min(batch_size, len(seq) - i)
    data_x = seq[i:i + seq_len]
    data_y = label[i:i + seq_len]

    input = torch.stack(torch.stack([item for item in data_x]).chunk(input_window, 1))
    target = torch.stack(torch.stack([item for item in data_y]).chunk(input_window, 1))
    return input, target

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        # return x + self.pe[:, :x.size(1), :]
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=512, num_layers=1,
                 in_channels=71, out_channels=1,dropout=0):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.ffn = nn.Linear(in_channels, out_channels)
        # self.ffn=nn.Linear(71,1)
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)

        output = output.transpose(0, 2)  # 1x50x71
        output = self.ffn(output)

        return output

def train(model, optimizer, scheduler, epoch, train_seq,train_label):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, len(train_seq) , batch_size)): # 注意到上述模型是step=batchsize
        data, targets = get_batch(train_seq,train_label,i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = int(len(train_seq) / batch_size/5 )
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(
                epoch, batch, len(train_seq) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss))  # , math.exp(cur_loss)
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source_x,data_source_y):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 100
    result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source_x) - 1, eval_batch_size):
            data, targets= get_batch(data_source_x,data_source_y,i, eval_batch_size)
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss +=  len(data)*criterion(output, targets).cpu().item()
            else:
                total_loss +=  len(data)*criterion(output[-output_window:], targets[-output_window:]).cpu().item()
            result = torch.cat((result, output[-1].squeeze(1).view(-1).cpu()),
                                    0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, targets[-1].squeeze(1).view(-1).cpu()), 0)
    return total_loss / len(data_source_x),result,truth


if __name__ == '__main__':
    X, Y = get_bls_data(data_path='../data/multi_bls_data/', type='rossler',
                        predict_fature='x', predict_step=1,
                        is_split=False)
    # X_train_bls, X_test_bls, train_y, test_y = get_bls_data(data_path='../data/multi_bls_data/', type='lorenz',
    #                                                         predict_fature='x', predict_step=1,
    #                                                         is_split=True)
    # 单维预测
    X_broad = X[0] # [3935,71]
    train_seq, train_label, \
        valid_seq, valid_label, \
        test_seq, test_label = get_data(X_broad, Y)
    # input, target = get_batch(test_seq, test_label, 900, batch_size)
    # print(f'input shape:{input.shape}')
    # print(f'target shape:{input.shape}')
    model = TransAm().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.96)
    # input, target = get_batch(train_seq, train_label, 0, batch_size)
    best_val_loss = 1000
    best_test_loss = 1000

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model, optimizer, scheduler, epoch,train_seq, train_label)
        train_loss, train_output, train_target = evaluate(model, train_seq, train_label)
        valid_loss, valid_output, valid_target = evaluate(model, valid_seq, valid_label)
        test_loss, test_output, test_target = evaluate(model, test_seq, test_label)
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












