class MMTLayer(nn.Module):
    def __init__(self, input_dim, rank, n_modals, beta, droprate=0.1) -> None:
        super(MMTLayer, self).__init__()
        self.input_dim = input_dim
        self.n_modals = n_modals

        self.attention = Attention(input_dim, rank, n_modals, beta)
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        aware = self.attention(x)
        aware = [self.dropout(aware[j]) for j in range(self.n_modals)]
        return aware



class Attention(nn.Module):
    def __init__(self, input_dim, rank, n_modals, beta) -> None:
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.n_modals = n_modals
        self.beta = beta

        self.trans_q1 = self.get_trans()
        self.trans_q2 = self.get_trans()
        self.trans_k1 = self.get_trans()
        self.trans_k2 = self.get_trans()
        self.lin_att = nn.ModuleList([Linear(rank * rank, input_dim[j]) for j in range(n_modals)])
        
        
    def get_trans(self):
        return nn.ModuleList([
            Linear(self.input_dim[j], self.rank) 
                for j in range(self.n_modals)
        ])
    
    def forward(self, x):
        """
        Input: List[torch.Tensor[batch_size, embed_dim]]
        """
        G_qk = []
        M_qk = []
        att = []
        for j in range(self.n_modals):
            G_q = self.trans_q1[j](x[j]).unsqueeze(-1) * self.trans_q2[j](x[j]).unsqueeze(-2) # mode-1 khatri-rao product
            G_k = self.trans_k1[j](x[j]).unsqueeze(-1) * self.trans_k2[j](x[j]).unsqueeze(-2)
            G_qk.append(G_q * G_k)
            M_qk.append(G_qk[j].mean(dim=0))
        
            
        for j in range(self.n_modals):
            att.append(G_qk[j])
            for l in range(self.n_modals):
                if j == l: continue
                att[j] = torch.einsum('ikl, lo->iko' ,att[j], M_qk[l]) # Tensor contraction
            B, R1, R2 = att[j].size()
            att[j] = att[j].view(B, R1 * R2)
            att[j] = self.lin_att[j](att[j])
            _att = att
            att[j] = att[j] * x[j] + self.beta * x[j]

        return att

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.lin = nn.Linear(in_features, out_features, bias)
    
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias:
            nn.init.constant_(self.lin.bias, 0.)
    
    def forward(self, x):
        return self.lin(x)
    

class MMTFusion(nn.Module): 
    def __init__(self, input_dim=[768, 1024], output_dim=6, nlayers=1, rank=8, beta=0.1, droprate=0.1):
        super(MMTFusion, self).__init__()
        
        self.layers = nn.ModuleList([MMTLayer(input_dim, rank, 2, beta, droprate) 
                                    for _ in range(nlayers)])
        
        self.fc = Linear(sum(input_dim), output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, a, v):
        
        out = [a, v]
        for layer in self.layers: 
            out = layer(out)
        
        out = self.fc(torch.cat(out, dim=1))        
        return a, v, out