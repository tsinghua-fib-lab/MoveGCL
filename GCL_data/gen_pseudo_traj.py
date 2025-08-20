import os
import numpy as np
import ast
import torch
from torch.nn import functional as F
import random
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import re
import random

from base_model import Traj_Config,Traj_Model 
from updated_model import Traj_Incremental_Config,Traj_Model_Incremental
import argparse
parser = argparse.ArgumentParser(description="traj_MOE_MODEL")
parser.add_argument('--city', default='Atlanta', help='City to sample')
parser.add_argument('--teacher_model', 
                    default="./train_result/incremental_step1/20_final_0608/test_['WashingtonDC', 'Atlanta', 'LosAngeles', 'NewYork']_Increm['Seattle']_6_512/test_froze_bottom_all_stage_unfroze_froze_top1_distill_experts_froze[[0, 1, 2, 3, 4], [0, 1, 3], [0, 1, 3], [2, 3], [1, 2, 3], [0, 1]]", 
                    help="teacher model's dir")
args = parser.parse_args()
city_name=args.city

batch_size=1000

log_dir = args.teacher_model
log_settings_path=log_dir+"/log_settings.txt"
replaced_first_loc_data_root=f"./GCL_data/replaced_first_loc_data/{city_name}"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def is_hex(s):
    # 检查字符串是否是十六进制
    return bool(re.match(r'^[0-9a-fA-F]+$', s))

def load_log_settings(file_path):
    """
    从 log_settings.txt 文件中读取参数，并返回一个字典
    """
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split(': ', 1)
                params[key] = value
    return params
def convert_value(value):
    """
    将字符串值转换为相应的数字类型（int 或 float），如果可能
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
        
def add_params_to_parser(parser, params):
    """
    将参数添加到 argparse.ArgumentParser 中，并转换数字类型
    """
    for key, value in params.items():
        if key not in ['device', 'seed']:
            converted_value = convert_value(value)
            parser.add_argument(f'--{key}', default=converted_value, type=type(converted_value), help=f'{key} from log_settings.txt')

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)                   
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrajDataset(Dataset):
    def __init__(self, data_root, split, B,T,few_shot,LPR=False):
        self.B = B
        self.T = T
        self.split = split
        self.few_shot = few_shot
        
        # load the shards
        shards = os.listdir(data_root)
        shards = [s for s in shards if any(x in s for x in split)]
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        
        self.data_city = defaultdict()
        for shard in self.shards:
            self.data_city[shard] = self.load_traj(shard)
        self.data = []

        
        batches = {shard: [self.data_city[shard][i:i + self.B] 
                           for i in range(0, int(len(self.data_city[shard])//self.B)*self.B, self.B)] 
                   for shard in self.shards}


        total_batches = sum(len(batches[shard]) for shard in self.shards)
        print(total_batches)


        shard_indices = {shard: 0 for shard in self.shards}


        remaining_batches = {shard: len(batches[shard]) for shard in self.shards}
        print(remaining_batches)

        if LPR:
            # print(111)
            for shard in self.shards:
                for _ in range(len(batches[shard])):
                    batch = batches[shard][shard_indices[shard]]
                    self.data.extend(batch)

                    shard_indices[shard] += 1
                    remaining_batches[shard] -= 1
        
        else:
            self.data_list=[]
            for shard in self.shards:
                for _ in range(len(batches[shard])):
                    # print("a")
                    batch = batches[shard][shard_indices[shard]]
                    self.data_list.append(batch)

                    shard_indices[shard] += 1
                    remaining_batches[shard] -= 1
            # random.shuffle(self.data_list)
            for trajs in self.data_list:
                self.data.extend(trajs)

                
    def load_traj(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            data = []
            for line in lines:
                traj = []
                line = line.strip()
                userid = line.split(' ')[0]
                rg=line.split(' ')[1]
                entropy=line.split(' ')[2]
                trajs = line.split(' ')[3]
                parts = trajs.strip().split(';')

                if is_hex(userid):
                    userid_decimal = str(int(userid, 16))
                else:
                    userid_decimal = userid  # 如果不是十六进制数，保持原样
                
                for part in parts:
                    if part:  
                        location, day, time, delta_time, quantized_distance = part.split(',')
                        day = int(day)
                        time = int(time)
                        delta_time=int(delta_time)
                        quantized_distance= int(quantized_distance)
                        rg=int(rg)
                        entropy=int(entropy)
                        traj.append([int(location) + 2, time,delta_time,quantized_distance,rg,entropy,int(userid_decimal[:4]),day])

                traj.append([int(1), int(0), int(0), int(0), int(0), int(0), int(0), int(0)])
                for _ in range(self.T+1 - len(traj)):
                    traj.append([int(0), int(0), int(0), int(0), int(0), int(0), int(0), int(0)])
                # print(traj)
                traj = torch.tensor(traj, dtype=torch.long)
                data.append([traj,filename.split('/')[-1].split('_')[0]])
            if self.few_shot:
                length = int(self.few_shot*len(data))
                data = data[:length]
            return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj,file = self.data[idx]             
        x = traj[:-1, 0]
        y = traj[1:, 0]
        ts_his = traj[:-1, 1]
        delta_t_his=traj[:-1, 2]
        delta_dis_his=traj[:-1,3]
        rg=traj[:-1,4]
        entropy=traj[:-1,5]
        user_id=traj[:-1,6]
        day=traj[:-1,7]
        return x, y, ts_his,delta_t_his,delta_dis_his,rg,entropy,user_id,day,file


def get_dataloader(data_root, split, B, T,few_shot,LPR=False):
    dataset = TrajDataset(data_root, split, B,T,few_shot,LPR=LPR)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=B, shuffle=False)
    return dataloader



def evaluate(model,config,test_loader,log_dir,B,city,device):
    file_to_write = config.write_dir + f"/{city[0]}_gen.txt"
    with open(file_to_write, "w") as f: # open for writing to clear the file
        pass
    model.eval()
    val_loss_accum = 0.0
    gate_city={}
    gate_city_count = {}

    for city1 in city:
        gate_city[city1]={}
        gate_city_count[city1] = {}
        for idx in range(config.n_layer):
            if hasattr(config, 'add_exp_num'):
                gate_city[city1][idx]=torch.zeros(config.num_experts+config.add_exp_num).to(config.device)
                gate_city_count[city1][idx] = torch.zeros(config.num_experts + config.add_exp_num).to(device)
            else:
                gate_city[city1][idx]=torch.zeros(config.num_experts).to(config.device)
                gate_city_count[city1][idx] = torch.zeros(config.num_experts).to(device)
    
    with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
        target_prob=[]
        for batch_no, test_batch in enumerate(it, start=1):
            x_test = test_batch[0]
            # print(f"x_test:{x_test.size()}")
            # print(x_test)
            y_test = test_batch[1]
            ts = test_batch[2]
            delta_ts = test_batch[3]
            delta_dis_his = test_batch[4]
            rg = test_batch[5]
            entropy= test_batch[6]
            user_id=test_batch[7][:,0]
            day=test_batch[8]
            test_city = test_batch[9][0]

            vocab = np.load(f'./location_feature/vocab_{test_city}.npy')
            vocab = np.pad(vocab, ((2,0), (0, 0)), mode='constant', constant_values=0)
            vocab = torch.from_numpy(vocab)
            vocab = vocab.to(torch.float32)
            device_type = "cuda" if device.startswith("cuda") else "cpu"
            seq_len = x_test.size(1)
            x_test = x_test.clone()  # 克隆防止修改原始输入
            top1_preds = torch.zeros_like(x_test)  # 存储每个时刻的top1预测
            # 掩码：哪些序列已经“终止”，默认都未终止
            terminated_mask = torch.zeros(x_test.size(0), dtype=torch.bool)

            # 初始化一个列表用于存储当前 batch 所有样本的 soft targets
            soft_targets_batch = torch.zeros(x_test.size(0), seq_len, vocab.size(0), device=device)  # [B, T, vocab_size]


            with torch.no_grad():
                for idx in range(1, seq_len):  # 从第1步开始预测（第0个点为已知）
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        output, gate_all = model(
                            x_test,
                            ts,
                            y_test,
                            delta_ts,
                            delta_dis_his,
                            rg,
                            entropy,
                            vocab,
                            test_city,
                            device
                        )

                        pred = output["logits"]  # [B, T, vocab_size]
                        pred[:, :, 0] = float('-inf') 

                        logits = pred[:, idx - 1]
                        topk_vals, topk_idxs = torch.topk(logits, k=5, dim=-1)  # [B, 5]

                        # 从 top-5 中随机选一个
                        rand_idx = torch.randint(0, 5, (logits.size(0),), device=logits.device)  # [B]
                        sampled_idx = topk_idxs[torch.arange(logits.size(0)), rand_idx]  # [B]

                        sampled_idx = sampled_idx.to(x_test.device)

                        # 只更新未终止序列的 soft targets
                        not_terminated = ~terminated_mask

                        current_token = x_test[:, idx]
                        newly_terminated = (current_token == 1) & (~terminated_mask)
                        terminated_mask |= newly_terminated 

                        # 只更新那些没有终止的序列
                        not_terminated = ~terminated_mask
                        x_test[not_terminated, idx] = sampled_idx[not_terminated]
                        top1_preds[not_terminated, idx] = sampled_idx[not_terminated]
            # 将数据写回文件
            file_to_write = config.write_dir + f"/{test_city}_gen.txt"

            with open(file_to_write, 'a', encoding='utf-8') as f:
                for b in range(x_test.size(0)):
                    # 找到结束符位置
                    end_idx = (x_test[b] == 1).nonzero(as_tuple=True)[0]
                    if end_idx.size(0) > 0:
                        traj_len = end_idx[0].item()  # 写入结束符前的内容
                    else:
                        traj_len = x_test.size(1)

                    user = user_id[b].item()
                    rg_b = rg[b][0].item()
                    entropy_b = entropy[b][0].item()

                    traj_parts = []
                    for t in range(traj_len):
                        loc = x_test[b, t].item()
                        d = day[b, t].item()
                        ti = ts[b, t].item()
                        delta_t = delta_ts[b, t].item()
                        delta_d = delta_dis_his[b, t].item()
                        traj_parts.append(f"{loc-2},{d},{ti},{delta_t},{delta_d}")

                    traj_line = f"{user} {rg_b} {entropy_b} {';'.join(traj_parts)}\n"
                    f.write(traj_line)

            print(f"Batch {batch_no} processed, data written to {file_to_write}")
            

params = load_log_settings(log_settings_path)

parser = argparse.ArgumentParser(description="traj_MOE_MODEL")

parser.add_argument('--write_dir', default=f'./GCL_data/pseudo_traj')

parser.add_argument('--device', default='cuda', help='Device for Attack')

# 解析参数
add_params_to_parser(parser, params)
args = parser.parse_args()

args.test_root=replaced_first_loc_data_root
args.B=batch_size

last_dir = os.path.basename(log_dir)

# 打印参数
print(args)

if hasattr(args, 'city_Incerm') and hasattr(args, 'city_original'):
    city_test = args.city_original.copy() 
    city_test.extend(args.city_Incerm)
    os.makedirs(args.write_dir+f"/{city_name}_{city_test}_{last_dir}", exist_ok=True)
    args.write_dir=args.write_dir+f"/{city_name}_{city_test}_{last_dir}"
    model = Traj_Model(Traj_Config(n_embd=args.n_embd,
                            n_head=args.n_head,
                            n_layer=args.n_layer,
                            num_experts=args.num_experts+args.add_exp_num,
                            top_k=args.top_k,
                            target_city=city_test)).to(args.device)
    model.load_state_dict(torch.load(log_dir + f"/model_{args.city_Incerm[0]}.pth",weights_only=True))
    model.eval()
    for city in city_test:
        city = [f'{city}']
        test_loader = get_dataloader(args.test_root, 
                                     city, 
                                     args.B, 
                                     args.T,
                                     few_shot=False,
                                     LPR=True)
        evaluate(model,args,test_loader=test_loader,log_dir=log_dir,B=args.B,city=city,device=args.device)
    
    
else:
    os.makedirs(args.write_dir+f"/{city_name}_{args.target_city}_{last_dir}", exist_ok=True)
    args.write_dir=args.write_dir+f"/{city_name}_{args.target_city}_{last_dir}"
    for city in args.target_city:
        model = Traj_Model(Traj_Config(n_embd=args.n_embd,
                                    n_head=args.n_head,
                                    n_layer=args.n_layer,
                                    num_experts=args.num_experts,
                                    top_k=args.top_k,
                                    target_city=args.target_city
                                    )).to(args.device)
        city = [f'{city}']
        model.load_state_dict(torch.load(log_dir + f"/model_{city[0]}.pth",weights_only=True))
        test_loader = get_dataloader(args.test_root, 
                                     city, 
                                     args.B, 
                                     args.T,
                                     few_shot=False,
                                     LPR=True)
        model.eval()
        evaluate(model,args,test_loader=test_loader,log_dir=log_dir,B=args.B,city=city,device=args.device)

