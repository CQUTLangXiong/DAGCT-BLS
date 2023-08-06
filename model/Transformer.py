import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
from embedding import DataEmbedding

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    @property
    def mask(self):
        return self._mask

class Attention(nn.Module):
    '''
    add spatial
    '''
    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.1, output_attention=False):
        super(Attention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask): # B, N, L, H, E
        # B, L, H, E = queries.shape
        B, N, L, H, E = queries.shape
        # _, S, _, D = values.shape
        _, _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        scores = torch.einsum('bnlhe, bnshe->bnhls', queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = self.dropout(torch.softmax(scale * scores, dim=-1)) # [B,N,H,L,L]
        # V = torch.einsum("bhls,bshd->blhd", A, values)
        V = torch.einsum("bnhls,bnshd->bnlhd", attn, values)

        if self.output_attention:
            return (V.contiguous(), attn)
        else:
            return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        # B, L, _ = queries.shape
        # _, S, _ = keys.shape
        B, N, L,_ = queries.shape
        _, _, S, _ = keys.shape
        H = self.n_heads

        # queries = self.query_projection(queries).view(B, L, H, -1) # [B,  L, 8,64)
        queries = self.query_projection(queries).view(B, N, L, H, -1) # [B,N,L,8,64]
        # keys = self.key_projection(keys).view(B, S, H, -1)
        keys = self.key_projection(keys).view(B, N, S, H, -1)
        # values = self.value_projection(values).view(B, S, H, -1)
        values = self.value_projection(values).view(B, N, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        # out-->[64,n,12,8,64]-->[64,12,512]
        # out = out.view(B, L, -1)
        out = out.view(B, N, L, -1)
        return self.out_projection(out), attn

class EncoderLayer(nn.Module):
    """
    encoder layer
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None): # x-->[b,n,l,c]
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x) # 残差连接--[b,n,l,c]
        x = self.norm1(x)
        y = x  # [b,n,l,c]-->[64,3,100,512]
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1)))) # 前馈神经网络-->[64, 4*512, 12]
        y = self.dropout(self.activation(self.conv1(y.permute(0,3,2,1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))  # [84,512,12]-->[64,12,512]
        y = self.dropout(self.conv2(y).permute(0,3,2,1))
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class Transformer(nn.Module):
    '''
    生成预测不需要对time_steps进行变形
    '''
    def __init__(self, in_channels=71, out_channels=1,d_model=512,
                 n_heads=8, dropout=0.1, num_layers=1, output_attention=True):
        super(Transformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(d_model)

        self.embedding = DataEmbedding(in_channels, d_model, dropout)
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    Attention(mask_flag=False, scale=None, attention_dropout=dropout, output_attention=output_attention),
                    d_model, n_heads,
                    d_keys=None, d_values=None
                ),
                d_model=d_model,d_ff=None, dropout=dropout, activation="relu"
            ) for _ in range(num_layers)
        ])
        self.decoder = nn.Linear(d_model, out_channels)

    def forward(self, x): # [100,3,1,71]
        # embedding
        # x = x.permute(2,1,0,3) # [1,3,100,71]
        enc_in = self.embedding(x) # [100,3,100,512]
        enc_out, attns = self.encoder(enc_in, attn_mask=None) #
        # 是否norm以下
        enc_out = self.norm(enc_out)

        dec_out = self.decoder(enc_out) # [100,3,100,1]
        # 只预测x
        dec_out = dec_out[:,0,:,:] # [100,100,1]
        return dec_out, attns


if __name__ == '__main__':
    # x = torch.rand(100, 3, 100, 512)
    # attention = Attention(mask_flag=False, scale=None, attention_dropout=0.1, output_attention=True)
    # attention_l = AttentionLayer(attention, d_model=512, n_heads=8,
    #                                    d_keys=None, d_values=None)
    # out, attn = attention_l(x, x, x, attn_mask=None)
    # encoder_layer = EncoderLayer(attention_l, d_model=512,
    #                              d_ff=None, dropout=0.1, activation="relu")
    # encoder_layers = [encoder_layer for l in range(3)]
    # encoder = Encoder(encoder_layers, conv_layers=None, norm_layer=None)
    # out1, attns = encoder(x, attn_mask=None)
    # print(out.shape)
    # print(out1.shape)

    x = torch.rand(100, 3, 1, 71)
    model = Transformer(in_channels=71, out_channels=1,d_model=512,
                        n_heads=8, dropout=0.1, num_layers=1, output_attention=True)
    out2, attns = model(x)
    print(out2.shape)