import torch
import torch.nn as nn

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = self.weights[i](x)
            x = x0 * xw + self.biases[i] + x
        return x

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class DCN(nn.Module):
    def __init__(self, config, num_cross_layers):
        super().__init__()
        self.cross_network = CrossNetwork(config.n_embd, num_cross_layers)
        self.deep_network = DeepNetwork(config.n_embd, 4 * config.n_embd, config.n_embd)
        self.combined_output = nn.Linear(config.n_embd * 2, config.n_embd)

    def forward(self, x):
        cross_out = self.cross_network(x)
        deep_out = self.deep_network(x)
        combined = torch.cat([cross_out, deep_out], dim=-1)
        output = self.combined_output(combined)
        return output

class Location_Tower(nn.Module):
    def __init__(self, config, num_cross_layers=2):
        super().__init__()
        self.lon_lat_embedding = nn.Linear(2, config.n_embd // 2)
        self.poi_feature_embedding = nn.Linear(34*2, config.n_embd // 4)
        # self.poi_feature_embedding = nn.Linear(14*2, config.n_embd // 4)
        self.flow_rank_embedding = nn.Embedding(9, config.n_embd // 4)
        self.dcn = DCN(config, num_cross_layers)

    def forward(self, vocab,visual_DNC=None):
        vocab_poi = vocab[:, :34*2]
        vocab_lon_lat = vocab[:, 34*2:34*2+2]
        # vocab_poi = vocab[:, :14*2]
        # vocab_lon_lat = vocab[:, 14*2:14*2+2]
        vocab_rank = vocab[:, -1].to(torch.long)
        
        vocab_poi_embedding = self.poi_feature_embedding(vocab_poi)  # Shape: [batch_size, n_embd // 4]
        vocab_lon_lat_emb = self.lon_lat_embedding(vocab_lon_lat)  # Shape: [batch_size, n_embd // 2]
        vocab_rank_emb = self.flow_rank_embedding(vocab_rank)  # Shape: [batch_size, n_embd // 4]
        
        vocab_embedding0 = torch.cat((vocab_lon_lat_emb, vocab_rank_emb,vocab_poi_embedding), dim=-1) 
        if visual_DNC:
            input=vocab_embedding0
            vocab_embedding = self.dcn(vocab_embedding0)
            output=vocab_embedding
            return vocab_embedding,input,output
        else:
            vocab_embedding = self.dcn(vocab_embedding0)
            return vocab_embedding