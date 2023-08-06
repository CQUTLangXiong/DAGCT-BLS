import torch
import torch.nn as nn
import math

# class PositionalEmbedding(nn.Module):
#
#     def __init__(self, d_model, max_len=100):
#         super(PositionalEmbedding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         # pe = pe.unsqueeze(0).transpose(0, 1)
#         pe = pe.unsqueeze(0)
#         # pe.requires_grad = False
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return x + self.pe[:,:x.size(1), :]
#         # return x + self.pe[:x.size(0), :]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x): # [64, 307, 12, 3]
        return self.pe[:, :x.size(2)] # [12,512]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=3, padding=padding, padding_mode='circular', bias=False)# [d_model,l]
        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular',
                                   bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x): # [64, 307, 12, 3]
        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        x = x.permute(0, 3, 2, 1) # [64, 3, 12, 307] # 307是高，宽为12
        x = self.tokenConv(x).permute(0, 3, 2, 1) # [64, 307, 12, 512]
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model=512, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x): # [64, 307, 12, 3]
        value_emb = self.value_embedding(x) # [64, 307, 12, 512]
        pos_emb = self.position_embedding(x) # [12, 512]
        return self.dropout(value_emb+pos_emb)


if __name__ == "__main__":
    x = torch.rand(100, 3,100,71)
    # pe = PositionalEmbedding(d_model=512, max_len=100)
    embed = DataEmbedding(c_in=71, d_model=512, dropout=0.1)
    out = embed(x)
    print(out.shape)