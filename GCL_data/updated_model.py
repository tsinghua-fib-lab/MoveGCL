from dataclasses import dataclass
import torch.nn as nn
from collections import OrderedDict
import torch
from torch.nn import functional as F
import numpy as np
import inspect
from city_location_embd import Location_Tower
from torch.nn import MultiheadAttention,Transformer
import copy
from dataclasses import dataclass, field
import gc

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
        self.topkroute_linear = nn.Linear(config.n_embd, config.num_experts)
        # add noise
        self.noise_linear =nn.Linear(config.n_embd, config.num_experts)

    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)
        #Noise logits
        noise_logits = self.noise_linear(mh_output)
        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise
        gate1 = F.softmax(noisy_logits, dim=-1)
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices,gate1
    
class SparseMoE(nn.Module):
    def __init__(self, config):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(config)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.top_k = config.top_k

    def forward(self, x):
        gating_output, indices,gate1 = self.router(x)
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
    
    def forward(self, x, attn_mask,pad_mask):
        y = self.ln_1(x)
        attn_output, attn_weights = self.attn(
            y,y,y,pad_mask,True,attn_mask,True,True
        )
        x = x + attn_output
        x_,gate_out = self.smoe(self.ln_2(x))
        x = x + x_
        # print(gate_out.size())

        return x,gate_out

class SplitLMHead(nn.Module):
    def __init__(self, n_embd, size_original, size_new, pretrained_weight, pretrained_bias=None):
        super().__init__()
        self.weight_old = nn.Parameter(pretrained_weight.clone(), requires_grad=False)
        new_vocab_size = size_new - size_original
        weight_old_mean = self.weight_old.mean(dim=0)
        device = self.weight_old.device
        weight_new = weight_old_mean.unsqueeze(0).expand(new_vocab_size, n_embd)
        weight_new.to(device)
        self.weight_new = nn.Parameter(weight_new)
        
        self.has_bias = pretrained_bias is not None
        if self.has_bias:
            self.bias_old = nn.Parameter(pretrained_bias.clone(), requires_grad=False)
            bias_new = torch.zeros(new_vocab_size)
            self.bias_new = nn.Parameter(bias_new, requires_grad=True)
        else:
            self.bias_old = None
            self.bias_new = None

    def forward(self, x):
        weight = torch.cat([self.weight_old, self.weight_new], dim=0)
        if self.has_bias:
            bias = torch.cat([self.bias_old, self.bias_new], dim=0)
        else:
            bias = None
        return F.linear(x, weight, bias)
    
    def freeze_original_blocks(self):
        self.weight_old.requires_grad = False
        if self.has_bias:
            self.bias_old.requires_grad = False
        self.weight_new.requires_grad = True
        if self.has_bias:
            self.bias_new.requires_grad = True

class SparseMoE_Incremental_Learning(nn.Module):
    def __init__(self, config,pretrained_moe_block, add_exp_num,router_model="normal"):
        super(SparseMoE_Incremental_Learning, self).__init__()
        self.city_len=len(config.city_original)+len(config.city_Incerm)
        self.topk = pretrained_moe_block.top_k
        self.experts = nn.ModuleList(pretrained_moe_block.experts)
        self.router=copy.deepcopy(pretrained_moe_block.router)
        with torch.no_grad():
            avg_params = {}
            for name, param in self.experts[0].named_parameters():
                avg_params[name] = torch.mean(
                    torch.stack([expert.state_dict()[name] for expert in pretrained_moe_block.experts]), dim=0
                )
        for _ in range(add_exp_num):
            new_expert = copy.deepcopy(pretrained_moe_block.experts[0])
            with torch.no_grad():
                for name, param in new_expert.named_parameters():
                    param.copy_(avg_params[name])
            self.experts.append(new_expert)

        with torch.no_grad():
            self.router.city_embeddings = nn.Parameter(
                    torch.randn(self.city_len, 32, device=pretrained_moe_block.router.city_embeddings.device)
                )
            pretrained_embeddings = pretrained_moe_block.router.city_embeddings[:len(config.city_original)]
            self.router.city_embeddings[:len(config.city_original), :] = pretrained_embeddings.detach().clone()
            self.router.city_to_index= {city: idx for idx, city in enumerate(config.city_target)}

            print(self.router.city_to_index)

            linear_size=int(config.n_embd + 32+config.n_embd/4+config.n_embd/4+config.n_embd/8+config.n_embd/8)
            self.router.topkroute_linear = nn.Linear(in_features=linear_size, out_features=config.num_experts + add_exp_num)
            
            self.router.topkroute_linear.weight[:config.num_experts,:] = pretrained_moe_block.router.topkroute_linear.weight
            weight_old_mean = pretrained_moe_block.router.topkroute_linear.weight.mean(dim=0)
            self.router.topkroute_linear.weight[config.num_experts:,:] = weight_old_mean.unsqueeze(0).repeat(add_exp_num, 1)
            
            has_bias = pretrained_moe_block.router.topkroute_linear.bias is not None
            if has_bias:
                self.router.topkroute_linear.bias[:config.num_experts] = pretrained_moe_block.router.topkroute_linear.bias
                bias_old_mean = pretrained_moe_block.router.topkroute_linear.bias.mean(dim=0)
                self.router.topkroute_linear.bias[config.num_experts:] = bias_old_mean.unsqueeze(0).repeat(add_exp_num)

            else:
                self.router.topkroute_linear.bias = None
            
            self.router.noise_linear = nn.Linear(in_features=linear_size, out_features=config.num_experts + add_exp_num)
            
            self.router.noise_linear.weight[:config.num_experts,:] = pretrained_moe_block.router.noise_linear.weight
            weight_old_mean = pretrained_moe_block.router.noise_linear.weight.mean(dim=0)
            self.router.noise_linear.weight[config.num_experts:,:] = weight_old_mean.unsqueeze(0).repeat(add_exp_num, 1)
            
            has_bias = pretrained_moe_block.router.noise_linear.bias is not None
            if has_bias:
                self.router.noise_linear.bias[:config.num_experts] = pretrained_moe_block.router.noise_linear.bias
                bias_old_mean = pretrained_moe_block.router.noise_linear.bias.mean(dim=0)
                self.router.noise_linear.bias[config.num_experts:] = bias_old_mean.unsqueeze(0).repeat(add_exp_num)
            else:
                self.router.noise_linear.bias = None


    
    def forward(self, x,city,delta_t_info,delta_dis_info,delta_rg_info,delta_entropy_info):
        gating_output, indices,gate1 = self.router(x,city,delta_t_info,delta_dis_info,delta_rg_info,delta_entropy_info)
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
    
    
@dataclass
class Traj_Incremental_Config:
    block_size: int = 48*3 # max seq_len
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512 # embedding dim
    num_experts: int = 8
    top_k: int = 2
    add_exp_num: int=1
    idx_add_exp: list = field(default_factory=lambda: None)
    experts_froze: list = field(default_factory=lambda: None)
    vocab_embed_froze: bool=True
    pos_tim_embed_froze: bool=True
    different_lr: bool=False
    router_model: str="normal"
    add_block: bool=False
    add_block_place:str="bottom"
    city_original:list=None
    city_Incerm:list=None
    city_target:list=None




class Traj_Model_Incremental(nn.Module):
    def __init__(self, config,model=None):
        super().__init__()
        self.config = config
        if model is None:
            raise ValueError("Pretrained model must be provided for incremental learning.")
    
        self.transformer = nn.ModuleDict(dict(
            time_embedding = copy.deepcopy(model.transformer.time_embedding),
            lon_lat_embedding = copy.deepcopy(model.transformer.lon_lat_embedding),
            poi_feature_embedding = copy.deepcopy(model.transformer.poi_feature_embedding),
            flow_rank_embedding = copy.deepcopy(model.transformer.flow_rank_embedding),
            wpe = copy.deepcopy(model.transformer.wpe),
            ln_f = copy.deepcopy(model.transformer.ln_f)
        ))
        self.vocab_embd = copy.deepcopy(model.vocab_embd)
        self.lm_head = copy.deepcopy(model.lm_head)
        self.transformer.h=nn.ModuleList([
            self._build_layer_with_optional_block(config, model.transformer.h, i)
            for i in range(config.n_layer)
        ])
        
        self.delta_t=nn.ModuleDict(dict(
            t_embedding=copy.deepcopy(model.delta_t.t_embedding),
            gpt_delta_t=copy.deepcopy(model.delta_t.gpt_delta_t),
            ln_f = copy.deepcopy(model.delta_t.ln_f),
            lm_head = copy.deepcopy(model.delta_t.lm_head),
            vector_pool=copy.deepcopy(model.delta_t.vector_pool) 
        ))

        self.delta_dis=nn.ModuleDict(dict(
            dis_embedding=copy.deepcopy(model.delta_dis.dis_embedding),
            gpt_delta_dis=copy.deepcopy(model.delta_dis.gpt_delta_dis),
            ln_f = copy.deepcopy(model.delta_dis.ln_f),
            lm_head = copy.deepcopy(model.delta_dis.lm_head),
            vector_pool=copy.deepcopy(model.delta_dis.vector_pool) 
        ))

        self.delta_rg_embedding=copy.deepcopy(model.delta_rg_embedding)

        self.delta_entropy_embedding=copy.deepcopy(model.delta_entropy_embedding)

        del model
        gc.collect()
        torch.cuda.empty_cache()

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
                device):
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
        delta_rg_info=self.delta_rg_embedding(rg)
        delta_entropy_info=self.delta_entropy_embedding(entropy)


        mask = Transformer.generate_square_subsequent_mask(T,device=device).to(torch.bool)
    
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
            gate_all.append(gate)

          
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        loc_embedding = self.vocab_embd(vocab)

        logits = prob(x,loc_embedding)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),ignore_index=0)

        output = OrderedDict()

        output['logits'] = logits
        output['loss'] = loss

        return output,gate_all
    
    def _build_layer_with_optional_block(self, config, existing_layers, layer_idx):
        if layer_idx in config.idx_add_exp:
            new_layer=existing_layers[layer_idx]
            print(existing_layers[layer_idx].smoe.router.topkroute_linear.weight.size())
            new_layer.smoe=SparseMoE_Incremental_Learning(
            config=config,
            pretrained_moe_block=existing_layers[layer_idx].smoe, 
            add_exp_num=config.add_exp_num,
            router_model=config.router_model
            )
        else:
            new_layer=existing_layers[layer_idx]
        return new_layer

    
    def freeze_original_blocks(self,
                               router_model="normal",
                               epoch_part=0,
                               LPR=False,
                               stage_unfroze=False
                               ):
       
        for param in self.delta_dis.parameters():
            param.requires_grad = False
        for param in self.delta_t.parameters():
            param.requires_grad = False
        for param in self.delta_rg_embedding.parameters():
            param.requires_grad = False
        for param in self.delta_entropy_embedding.parameters():
            param.requires_grad = False

        if self.config.pos_tim_embed_froze:
            for param in self.transformer.time_embedding.parameters():
                param.requires_grad = False
            for param in self.transformer.lon_lat_embedding.parameters():
                param.requires_grad = False
            for param in self.transformer.poi_feature_embedding.parameters():
                param.requires_grad = False
            for param in self.transformer.flow_rank_embedding.parameters():
                param.requires_grad = False
            for param in self.transformer.wpe.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
        else:
            for param in self.transformer.time_embedding.parameters():
                param.requires_grad = True
            for param in self.transformer.lon_lat_embedding.parameters():
                param.requires_grad = True
            for param in self.transformer.poi_feature_embedding.parameters():
                param.requires_grad = True
            for param in self.transformer.flow_rank_embedding.parameters():
                param.requires_grad = True
            for param in self.transformer.wpe.parameters():
                param.requires_grad = True
            for param in self.lm_head.parameters():
                param.requires_grad = True

        # froze MoE transformers
        for idx, block in enumerate(self.transformer.h):
            for param in block.parameters():
                param.requires_grad = False
            if idx in self.config.idx_add_exp:
                if stage_unfroze:
                    if epoch_part<0.3 and idx in[1,2,3,4]:
                        continue
                    elif (epoch_part>=0.3 and epoch_part<0.6) and idx in [0,2,3,5]:
                        continue
                    elif epoch_part>=0.6 and idx in [0,1,4,5]:
                        continue

                for moe_idx,expert in enumerate(block.smoe.experts):
                    if moe_idx not in self.config.experts_froze[idx]:
                    # if moe_idx not in self.config.experts_froze:
                        for param in expert.parameters():
                            param.requires_grad = True
                for param in block.smoe.router.topkroute_linear.parameters():
                    param.requires_grad = True
                for param in block.smoe.router.noise_linear.parameters():
                    param.requires_grad = True
                block.smoe.router.city_embeddings.requires_grad = True
                                   
        for param in self.transformer.ln_f.parameters():
            param.requires_grad = False

        if self.config.vocab_embed_froze:
            for param in self.vocab_embd.parameters():
                param.requires_grad = False
    
        else:
            for param in self.vocab_embd.parameters():
                param.requires_grad = False
            
            for param in self.vocab_embd.lon_lat_embedding.parameters():
                param.requires_grad = True
            
            for param in self.vocab_embd.poi_feature_embedding.parameters():
                param.requires_grad = True
            
            for param in self.vocab_embd.flow_rank_embedding.parameters():
                param.requires_grad = True
            
            for param in self.vocab_embd.dcn.parameters():
                param.requires_grad = True

    
    def configure_optimizers(self, 
                             weight_decay, 
                             learning_rate,
                             epoch_part,
                             LPR_train=False,
                             stage_unfroze=False,):
        if self.config.different_lr==True:
            self.freeze_original_blocks(epoch_part=epoch_part,
                                        LPR=LPR_train,
                                        router_model=self.config.router_model,
                                        stage_unfroze=stage_unfroze)
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

            special_params = []
            regular_params = []

            for pn, p in param_dict.items():
                if "transformer.time_embedding" in pn or \
                "transformer.lon_lat_embedding" in pn or \
                "transformer.poi_feature_embedding" in pn or \
                "transformer.flow_rank_embedding" in pn or\
                "transformer.wpe"in pn or\
                "vocab_embd.dcn"in pn or\
                "vocab_embd.flow_rank_embedding"in pn or\
                "vocab_embd.poi_feature_embedding"in pn or\
                "vocab_embd.lon_lat_embedding" in pn:
                    special_params.append(p)  # 特殊部分，学习率为 learning_rate * 0.1
                else:
                    regular_params.append(p)  # 其他部分，学习率为 learning_rate

            # 创建优化器组
            optim_groups = [
                {"params": special_params, "weight_decay": weight_decay, "lr": learning_rate * 0.1},
                {"params": regular_params, "weight_decay": weight_decay, "lr": learning_rate},
            ]

            # 打印参数组信息
            num_special_params = sum(p.numel() for p in special_params)
            num_regular_params = sum(p.numel() for p in regular_params)
            print(f"num special parameter tensors: {len(special_params)}, with {num_special_params:,} params")
            print(f"num regular parameter tensors: {len(regular_params)}, with {num_regular_params:,} params")

            # 创建 AdamW 优化器
            fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and torch.cuda.is_available()
            print(f"using fused AdamW: {use_fused}")

            optimizer = torch.optim.AdamW(optim_groups, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        
        else:
            self.freeze_original_blocks(epoch_part=epoch_part,
                                        LPR=LPR_train,
                                        router_model=self.config.router_model,
                                        stage_unfroze=stage_unfroze)
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
