from dataclasses import dataclass
import torch.nn as nn
from collections import OrderedDict
import torch
from torch.nn import functional as F
import numpy as np
import inspect
from city_location_embd import Location_Tower
from torch.nn import MultiheadAttention,Transformer


@dataclass
class GPT_block_Config:
    block_size: int = 144 # max seq_len
    n_head: int = 4
    n_embd: int = 64 # embedding dim

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class GPT_Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttention(num_heads=config.n_head,embed_dim=config.n_embd,batch_first=True)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x, attn_mask,pad_mask):
        y = self.ln_1(x)
        attn_output, attn_weights = self.attn(
            y,y,y,pad_mask,True,attn_mask,True,True
        )
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        # print(attn_output)

        return x, attn_weights

def prob(x,vocab_embedding):
    x = x.unsqueeze(2)  # [B, T, 1, 512]
    vocab_embedding = vocab_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, loc_num, 512]
    cosine_similarity = torch.matmul(x, vocab_embedding.transpose(-1, -2))  # [B, T, 1, loc_num]
    cosine_similarity = cosine_similarity.squeeze(2)  # [B, T, loc_num]
    return cosine_similarity



class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class NoisyTopkRouter(nn.Module):
    def __init__(self, config):
        super(NoisyTopkRouter, self).__init__()
        
        self.top_k = config.top_k
        self.n_embd = config.n_embd
        self.num_experts = config.num_experts
        self.target_city = config.target_city
        self.city_to_index = {city: idx for idx, city in enumerate(self.target_city)}

        self.city_embeddings = nn.Parameter(torch.randn(len(self.target_city), 32))
        linear_size=int(self.n_embd + 32+self.n_embd/4+self.n_embd/4+self.n_embd/8+self.n_embd/8)
        self.topkroute_linear = nn.Linear(linear_size, self.num_experts)
        self.noise_linear = nn.Linear(linear_size, self.num_experts)
    
    def forward(self, 
                mh_output,
                city,
                delta_t_info,
                delta_dis_info,
                delta_rg_info,
                delta_entropy_info):
        batch_size, T, n_embd = mh_output.size()
        city_index = self.city_to_index[city]
        city_embed = self.city_embeddings[city_index]
        city_embeds = city_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, T, 32)

        mh_output = torch.cat([mh_output, 
                               city_embeds,
                               delta_t_info,
                               delta_dis_info,
                               delta_rg_info,
                               delta_entropy_info], dim=-1)
        
        logits = self.topkroute_linear(mh_output)
        
        if self.training:
            noise_logits = self.noise_linear(mh_output)
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        gate1 = F.softmax(noisy_logits, dim=-1)
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices, gate1
    
class SparseMoE(nn.Module):
    def __init__(self, config):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.top_k = config.top_k

    def forward(self, x,city,delta_t_info,delta_dis_info,delta_rg_info,delta_entropy_info):
        gating_output, indices,gate1 = self.router(x,
                                                   city,
                                                   delta_t_info,
                                                   delta_dis_info,
                                                   delta_rg_info,
                                                   delta_entropy_info)
        final_output = torch.zeros_like(x)
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output,gate1

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttention(num_heads=config.n_head,embed_dim=config.n_embd,batch_first=True)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.smoe = SparseMoE(config)
    
    def forward(self, 
                x, 
                attn_mask,
                pad_mask,
                city,
                delta_t_info,
                delta_dis_info,
                delta_rg_info,
                delta_entropy_info
                ):
        y = self.ln_1(x)
        attn_output, attn_weights = self.attn(
            y,y,y,pad_mask,True,attn_mask,True,True
        )
        x = x + attn_output
        x_,gate_out = self.smoe(self.ln_2(x),
                                city,
                                delta_t_info,
                                delta_dis_info,
                                delta_rg_info,
                                delta_entropy_info
                                )
        x = x + x_
        return x,gate_out

class VectorPoolModule(nn.Module):
    def __init__(self, n_embedding, pool_size=512):
        super().__init__()
        self.vector_pool = nn.Parameter(torch.randn(pool_size, n_embedding))
        self.scale = 1.0/(n_embedding ** 0.5)
        
    def forward(self, x):
        """
        x: [B, T, n_embedding]
        返回: [B, T, n_embedding]
        """
        B, T, C = x.shape
        x_flat = x.view(B*T, 1, C)
        
        dots = torch.matmul(x_flat, self.vector_pool.T.unsqueeze(0)) * self.scale
        
        weights = torch.softmax(dots, dim=-1)
        
        weighted_sum = torch.matmul(weights, self.vector_pool.unsqueeze(0))
        
        output = weighted_sum.view(B, T, C)
        
        return output


@dataclass
class Traj_Config:
    block_size: int = 48*3
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    num_experts: int = 8
    top_k: int = 2
    target_city: list=None



class Traj_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            time_embedding = nn.Embedding(48, config.n_embd),
            lon_lat_embedding = nn.Linear(2,config.n_embd//2),
            poi_feature_embedding = nn.Linear(34*2,config.n_embd//4),
            # poi_feature_embedding = nn.Linear(14*2,config.n_embd//4),
            flow_rank_embedding = nn.Embedding(9,config.n_embd//4),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # h = nn.ModuleList([Block_norm(config),Block_norm(config),Block(config),Block(config)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.vocab_embd = Location_Tower(config)
        self.lm_head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        delta_t_gpt_config=GPT_block_Config(block_size=config.block_size,
                                            n_head=config.n_head,
                                            n_embd=int(config.n_embd/4)
                                            )
        
        self.delta_t=nn.ModuleDict(dict(
            t_embedding=nn.Embedding(96,delta_t_gpt_config.n_embd),
            gpt_delta_t=nn.ModuleList([GPT_Block(delta_t_gpt_config) for _ in range(2)]),
            ln_f = nn.LayerNorm(delta_t_gpt_config.n_embd),
            lm_head = nn.Linear(delta_t_gpt_config.n_embd, delta_t_gpt_config.n_embd, bias=False),
            vector_pool=VectorPoolModule(delta_t_gpt_config.n_embd) 
        ))

        # self.delta_t_embedding=nn.Embedding(96,delta_t_gpt_config.n_embd)
        # self.gpt_delta_t=nn.ModuleList([GPT_Block(delta_t_gpt_config) for _ in range(2)])
        # self.gpt_delta_t=GPT_Block(delta_t_gpt_config)

        delta_dis_gpt_config=GPT_block_Config(block_size=config.block_size,
                                            n_head=config.n_head,
                                            n_embd=int(config.n_embd/4)
                                            )
        self.delta_dis=nn.ModuleDict(dict(
            dis_embedding=nn.Embedding(30,delta_dis_gpt_config.n_embd),
            gpt_delta_dis=nn.ModuleList([GPT_Block(delta_dis_gpt_config) for _ in range(2)]),
            ln_f = nn.LayerNorm(delta_dis_gpt_config.n_embd),
            lm_head = nn.Linear(delta_dis_gpt_config.n_embd, delta_dis_gpt_config.n_embd, bias=False),
            vector_pool=VectorPoolModule(delta_dis_gpt_config.n_embd) 
        ))

        # self.delta_dis_embedding=nn.Embedding(30,delta_dis_gpt_config.n_embd)
        # self.gpt_delta_dis=nn.ModuleList([GPT_Block(delta_dis_gpt_config) for _ in range(2)])
        # self.gpt_delta_dis=GPT_Block(delta_dis_gpt_config)

        delta_rg_gpt_config=GPT_block_Config(block_size=config.block_size,
                                            n_head=config.n_head,
                                            n_embd=int(config.n_embd/8)
                                            )
        self.delta_rg_embedding=nn.Embedding(30,delta_rg_gpt_config.n_embd)
        # self.gpt_delta_rg=nn.ModuleList([GPT_Block(delta_dis_gpt_config) for _ in range(2)])
        # self.gpt_delta_rg=GPT_Block(delta_rg_gpt_config)

        delta_entropy_gpt_config=GPT_block_Config(block_size=config.block_size,
                                            n_head=config.n_head,
                                            n_embd=int(config.n_embd/8)
                                            )
        self.delta_entropy_embedding=nn.Embedding(41,delta_entropy_gpt_config.n_embd)
        
        # init params
        self.apply(self._init_weights) # iterate all submodule and apply init_modules
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
                his, 
                ts ,
                targets,
                delta_ts,
                delta_dis_his,
                rg,
                entropy,
                vocab,
                city,
                device,
                visual_DNC=None):
        # for i in range(len(entropy)):
        #     print(entropy[i][0])
        # idx is of shape (B, T), T is time dimension
        # poi (B, T,25)
        gate_all=[]
        loc_feature=np.take(vocab, his, axis=0) 
        his, targets ,loc_feature,ts,delta_ts,delta_dis,rg,entropy= his.to(device), targets.to(device), loc_feature.to(device),ts.to(device),delta_ts.to(device),delta_dis_his.to(device),rg.to(device),entropy.to(device)
        B, T = his.size()
        padding_mask = (his==0).to(torch.bool)
        ts = ts.to(torch.long)
        delta_ts = delta_ts.to(torch.long)
        # delta_dis_his = delta_dis_his.to(torch.long)
        # rg=rg
        poi_feature = loc_feature[:,:,:34*2]
        lon_lat = loc_feature[:,:,34*2:34*2+2]
        # poi_feature = loc_feature[:,:,:14*2]
        # lon_lat = loc_feature[:,:,14*2:14*2+2]
        rank = loc_feature[:,:,-1].to(torch.long)
        vocab = vocab.to(device)
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=device) #shape (T)
        pos_emb = self.transformer.wpe(pos) 
        poi_feature_emb = self.transformer.poi_feature_embedding(poi_feature)
        lon_lat_emb = self.transformer.lon_lat_embedding(lon_lat)
        rank_emb = self.transformer.flow_rank_embedding(rank)
        token_emb = torch.cat((lon_lat_emb,rank_emb,poi_feature_emb),dim=-1)
        ts_emb = self.transformer.time_embedding(ts) #B T 16*3 
        x = token_emb + ts_emb + pos_emb 
        # print(f"delta_ts:{delta_ts}")
        # delta_t_info=self.lstm_detlta_t(delta_ts)
        # delta_t_embd=self.delta_t_embedding(delta_ts)
        # # delta_dis_his
        # delta_dis_embd=self.delta_dis_embedding(delta_dis_his)
        delta_rg_info=self.delta_rg_embedding(rg)
        delta_entropy_info=self.delta_entropy_embedding(entropy)

        mask = Transformer.generate_square_subsequent_mask(T,device=device).to(torch.bool)
        # for block
        # delta_t_info,_=self.gpt_delta_t(delta_t_embd,mask,padding_mask)
        # delta_dis_info,_=self.gpt_delta_dis(delta_dis_embd,mask,padding_mask)
        # delta_rg_info,_=self.gpt_delta_rg(delta_rg_embd,mask,padding_mask)
        delta_t_embd=self.delta_t.t_embedding(delta_ts)
        x_t=delta_t_embd
        for gpt_bock in  self.delta_t.gpt_delta_t:
            x_t,_=gpt_bock(
                x_t,mask,padding_mask
            )
        delta_t_info=self.delta_t.ln_f(x_t)
        delta_t_info=self.delta_t.lm_head(delta_t_info)
        delta_t_info=self.delta_t.vector_pool(delta_t_info)

        delta_dis_embd=self.delta_dis.dis_embedding(delta_dis)
        x_dis=delta_dis_embd
        for gpt_bock in  self.delta_dis.gpt_delta_dis:
            x_dis,_=gpt_bock(
                x_dis,mask,padding_mask
            )
        delta_dis_info=self.delta_t.ln_f(x_dis)
        delta_dis_info=self.delta_t.lm_head(delta_dis_info)
        delta_dis_info=self.delta_t.vector_pool(delta_dis_info)

        for block in self.transformer.h:
            x,gate= block(x, 
                            mask,
                            padding_mask,
                            city,
                            delta_t_info,
                            delta_dis_info,
                            delta_rg_info,
                            delta_entropy_info)
            # if isinstance(block, Block):
            #     x,gate= block(x, 
            #                 mask,
            #                 padding_mask,
            #                 city,
            #                 delta_t_info,
            #                 delta_dis_info,
            #                 delta_rg_info)
            # elif isinstance(block, Block_norm):  # 检查是否是Block_norm类
            #     x,gate= block(x, 
            #                 mask,
            #                 padding_mask,
            #                 )
            # else:
            #     print("Unknown block type")

            gate_all.append(gate)

          
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        if visual_DNC:
            loc_embedding,input_DNC,output_DNC = self.vocab_embd(vocab,visual_DNC=visual_DNC)
        else:
            loc_embedding = self.vocab_embd(vocab)

        logits = prob(x,loc_embedding)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),ignore_index=0)

        output = OrderedDict()

        output['logits'] = logits
        output['loss'] = loss

        if visual_DNC:
            return output,gate_all,input_DNC,output_DNC
        else:
            return output,gate_all
    

    
    def configure_optimizers(self, weight_decay, learning_rate,LPR_train=False):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} params")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} params")
        
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        if torch.cuda.is_available():
            device = "cuda"
        use_fused = fused_available and device == "cuda" ## 8. fuse the adamw
        print(f"using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        return optimizer
