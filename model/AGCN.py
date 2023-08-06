import torch
import torch.nn.functional as F
import torch.nn as nn


class DAGCN(nn.Module):
    def __init__(self, num_time_steps, num_nodes, in_dims, out_dims, cheb_k, embed_dim):
        '''
        embed_dim必须小于N
        '''
        super(DAGCN, self).__init__()
        self.num_time_steps = num_time_steps
        self.num_nodes = num_nodes
        self.cheb_k = cheb_k
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.embed_dim = embed_dim

        # 动态空间节点嵌入向量
        self.dn_embeddings = nn.Parameter(torch.randn(num_time_steps,
                                                      num_nodes,
                                                      embed_dim),
                                          requires_grad=True)    # [T, N, embed_dim]

        # Theta = E*W--->(tnd,d
        self.weights_pool = nn.Parameter(torch.randn(embed_dim,
                                                           cheb_k,
                                                           in_dims,
                                                           out_dims))
        self.bias_pool = nn.Parameter(torch.randn(embed_dim,
                                                        out_dims))

    def forward(self, x): # x-->[B,T,N,C]
        supports = F.softmax(F.relu(torch.einsum('tne, tse->tns', self.dn_embeddings, self.dn_embeddings)), dim=-1)
        unit = torch.stack([torch.eye(self.num_nodes).to(supports.device) for _ in range(self.num_time_steps)])
        support_set = [unit, supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.einsum('tnn, tns->tns', 2 * supports, support_set[-1])- support_set[-2])
        supports = torch.stack(support_set, dim=1) # [T, cheb_k, N,N]
        # theta
        theta = torch.einsum('tnd, dkio->tnkio', self.dn_embeddings, self.weights_pool) #T, N, cheb_k, dim_in, dim_out
        bias = torch.einsum('tnd, do->tno', self.dn_embeddings, self.bias_pool)  #T, N, dim_out
        x_g = torch.einsum('tknm, btmc->btknc', supports, x)
        x_gconv = torch.einsum('btkni, tnkio->btno', x_g, theta) + bias
        return x_gconv    # [B, T, N, dim_out]

if __name__ =='__main__':
    x = torch.rand(64, 100, 3, 71)
    model = DAGCN(num_time_steps=100, num_nodes=3, in_dims=71, out_dims=512, cheb_k=3, embed_dim=2)
    out = model(x)
    print(out.shape)


