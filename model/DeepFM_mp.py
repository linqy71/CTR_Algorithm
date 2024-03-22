import torch
import torch.nn as nn
import sys

class DeepFM(nn.Module):
    """
    """
    def create_emb(self, ln, dim, sparse=True, ndevices=4):
        embedding = nn.ModuleList()
        for i in range(0, len(ln)):
            n = ln[i]
            EE = nn.EmbeddingBag(n, dim, mode="sum" , sparse=sparse)
            torch.nn.init.xavier_uniform_(EE.weight.data)
            d = torch.device("cuda:" + str(i % ndevices))
            embedding.append(EE.to(d))

        return embedding
    
    def create_mlp(self, input_dim, mlp_dims, dropout):
        layers = nn.ModuleList()
        for mlp_dim in mlp_dims:
            LL = nn.Linear(input_dim, mlp_dim)
            nn.init.xavier_uniform_(LL.weight.data)
            nn.init.zeros_(LL.bias)
            layers.append(LL)
            layers.append(nn.BatchNorm1d(mlp_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = dropout))
            input_dim = mlp_dim
        bot_LL = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(bot_LL.weight.data)
        nn.init.zeros_(bot_LL.bias)
        layers.append(bot_LL)
        return nn.Sequential(*layers)
    
    def __init__(self, feature_fields, embed_dim, mlp_dims, dropout=0, ndevices=4):
        super(DeepFM, self).__init__()
        # self.offsets = np.array((0, *np.cumsum(feature_fields)[:-1]), dtype = np.long)
        self.ndevices = ndevices
        self.feature_fields = feature_fields
        
        self.linear = self.create_emb(feature_fields, 1)
        # self.linear = torch.nn.Embedding(sum(feature_fields)+1, 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,))).to("cuda:0")
        
        self.embedding = self.create_emb(feature_fields, embed_dim, ndevices=ndevices)

        self.embedding_out_dim = len(feature_fields) * embed_dim
        input_dim = self.embedding_out_dim
        
        self.mlp = self.create_mlp(input_dim, mlp_dims, dropout=0).to("cuda:0")
        
        
    
    def apply_emb(self, lS_i, linear=False):
        ly = []
        if linear:
            tables = self.linear
        else :
            tables = self.embedding
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = torch.zeros_like(sparse_index_group_batch)
            E = tables[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)
            ly.append(V)
        
        return ly
    
        
    def forward(self, x):
        """
        """ 
        # tmp = x + x.new_tensor(self.offsets).unsqueeze(0)
        x_list = []
        for k, _ in enumerate(self.embedding):
            d = torch.device("cuda:" + str(k % self.ndevices))
            x_list.append(x[:,k].to(d))

        ly = self.apply_emb(x_list, linear=False)
        if len(self.embedding) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")
        
        for i, emb in enumerate(ly):
            ly[i] = emb.to("cuda:0")
        embeddings = torch.stack(ly, dim=1)
 
        l_ly = self.apply_emb(x_list, linear=True)
        for i, emb in enumerate(l_ly):
            l_ly[i] = emb.to("cuda:0")
        l_ly = torch.stack(l_ly, dim=1)
        linear_part = torch.sum(l_ly, dim = 1) + self.bias
        
        square_of_sum = torch.sum(embeddings, dim=1) ** 2
        sum_of_square = torch.sum(embeddings ** 2, dim=1)
        inner_part = 0.5 * torch.sum(square_of_sum - sum_of_square, dim = 1, keepdim=True)
        
        fm_part = linear_part + inner_part
        
        mlp_part = self.mlp(embeddings.view(-1, self.embedding_out_dim))
        
        x = fm_part + mlp_part
        x = torch.sigmoid(x.squeeze(1))
        return x