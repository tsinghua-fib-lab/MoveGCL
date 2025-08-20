import os
import numpy as np
import ast
import torch
from torch.nn import functional as F
torch.cuda.empty_cache()  # Clear the cache
from dataloader_detla_lt import get_dataloader
import random
import argparse
import shutil
from tqdm import tqdm


from base_model import Traj_Config,Traj_Model 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
parser = argparse.ArgumentParser(description="traj_MOE_MODEL")
parser.add_argument('--model', 
                    default="./train_result/incremental_step1/20_final_0608/test_['WashingtonDC', 'Atlanta', 'LosAngeles', 'NewYork']_Increm['Seattle']_6_512/test_froze_bottom_all_stage_unfroze_froze_top1_distill_experts_froze[[0, 1, 2, 3, 4], [0, 1, 3], [0, 1, 3], [2, 3], [1, 2, 3], [0, 1]]", 
                    help="model path")
args = parser.parse_args()
log_dir = args.model
log_settings_path=log_dir+"/log_settings.txt"


def load_log_settings(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                key, value = line.split(': ', 1)
                params[key] = value
    return params
def convert_value(value):
    try:
        return ast.literal_eval(value)  # 尝试将字符串转换为 Python 对象
    except (ValueError, SyntaxError):
        try:
            return int(value)  # 尝试转换为整数
        except ValueError:
            try:
                return float(value)  # 尝试转换为浮点数
            except ValueError:
                return value  # 如果无法转换，返回原始字符串       
def add_params_to_parser(parser, params):
    for key, value in params.items():
        if key not in ['device', 'seed']:  # 跳过 device 和 seed
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

def evaluate(model,config,test_loader,log_dir,B,city,device,froze_info_file):
    log_file_test = os.path.join(froze_info_file, f"log_{city}_test.txt")
    with open(log_file_test, "w") as f: # open for writing to clear the file
        pass
    model.eval()
    acc1 = 0
    acc3 = 0
    acc5 = 0
    size = 0
    val_loss_accum = 0.0
    batch = 0
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
        for batch_no, test_batch in enumerate(it, start=1):
            with torch.no_grad():
                batch = batch_no+1
                x_test = test_batch[0]
                y_test = test_batch[1]
                ts = test_batch[2]
                delta_ts = test_batch[3]
                delta_dis_his = test_batch[4]
                rg = test_batch[5]
                entropy= test_batch[6]
                test_city = test_batch[7][0]

                vocab = np.load(f'./location_feature/vocab_{test_city}.npy')
                vocab = np.pad(vocab, ((2,0), (0, 0)), mode='constant', constant_values=0)
                vocab = torch.from_numpy(vocab)
                vocab = vocab.to(torch.float32)
                device_type = "cuda" if device.startswith("cuda") else "cpu"
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    output,gate_all = model(x_test,
                                            ts,
                                            y_test,
                                            delta_ts,
                                            delta_dis_his,
                                            rg,
                                            entropy,
                                            vocab,
                                            test_city,device)
                mask = (x_test > 1).unsqueeze(-1)
                mask = mask.to(device)
                for idx,gate_output in enumerate(gate_all):
                    masked_gate_output = gate_output * mask
                    sum_gate_output = torch.sum(masked_gate_output, dim=(0, 1))
                    gate_city[test_city][idx]+=sum_gate_output
                    count_gate_output = torch.sum(mask, dim=(0, 1))
                    gate_city_count[test_city][idx] += count_gate_output

                loss = output['loss'] 
                val_loss_accum += loss.detach()
                pred = output['logits']#[B T vocab_size]
                pred[:,:,0] = float('-inf')   
                y_test = y_test.to(device)
                for b in range(B):
                    if b >= pred.size(0):
                        break
                    _, pred_indices = torch.topk(pred[b], 100)
                    valid_mask = y_test[b] > 0
                    valid_y_val = y_test[b][valid_mask]
                    valid_pred_indices = pred_indices[valid_mask]
        
                    valid_y_val_expanded = valid_y_val.unsqueeze(1) 
                    l= valid_y_val_expanded.size(0)
                    size +=l
        
                    a1 = torch.sum(valid_pred_indices[:, 0:1] == valid_y_val_expanded).item()
                    a3 = torch.sum(valid_pred_indices[:, 0:3] == valid_y_val_expanded).item()
        
                    a5 = torch.sum(valid_pred_indices[:,0:5] == valid_y_val_expanded).item()
                    acc1 += a1
                    acc3 += a3
                    acc5 += a5


    val_loss_accum=val_loss_accum/ batch
    acc1 = acc1/size
    acc3 = acc3/size
    acc5 = acc5/size

    with open(log_file_test, "a") as f:
        f.write(f"{val_loss_accum}\t{acc1:.6f}\t{acc3:.6f}\t{acc5:.6f}\t{size}\n")   

    # Calculate average gate values
    gate_city_avg = {}
    for city_name in gate_city:
        gate_city_avg[city_name] = {}
        for layer_idx in gate_city[city_name]:
            # Avoid division by zero
            with torch.no_grad():
                avg = torch.zeros_like(gate_city[city_name][layer_idx])
                non_zero = gate_city_count[city_name][layer_idx] > 0
                avg[non_zero] = gate_city[city_name][layer_idx][non_zero] / gate_city_count[city_name][layer_idx][non_zero]
                gate_city_avg[city_name][layer_idx] = avg

    # Save average gate values to file
    gate_file = os.path.join(froze_info_file, f"gate_city_avg_{city[0]}.txt")
    with open(gate_file, "w") as f:
        for city_name, gate_dict in gate_city_avg.items():
            f.write(f"City: {city_name}\n")
            for idx, gate_tensor in gate_dict.items():
                gate_list = gate_tensor.cpu().tolist()
                f.write(f"Layer {idx}: {gate_list}\n")

    # 将 gate_city 保存到文件
    gate_file = os.path.join(froze_info_file, f"gate_city_{city[0]}.txt")
    with open(gate_file, "w") as f:
        for city_name, gate_dict in gate_city.items():
            f.write(f"City: {city_name}\n")
            for idx, gate_tensor in gate_dict.items():
                # 将张量从 GPU 移动到 CPU 并转换为列表
                gate_list = gate_tensor.cpu().tolist()
                f.write(f"Layer {idx}: {gate_list}\n")

params = load_log_settings(log_settings_path)
parser = argparse.ArgumentParser(description="traj_MOE_MODEL")
parser.add_argument('--device', default='cuda', help='Device for Attack')

add_params_to_parser(parser, params)

frozen_test_file="./frozen_test_file"
# 确保 frozen_test_file 是一个已存在的目录
if os.path.exists(frozen_test_file):
    # 清空目录
    for f in os.listdir(frozen_test_file):
        f_path = os.path.join(frozen_test_file, f)
        if os.path.isfile(f_path):
            os.remove(f_path)
        elif os.path.isdir(f_path):
            shutil.rmtree(f_path)
else:
    os.makedirs(frozen_test_file)

# 解析参数
args = parser.parse_args()
args.B=128
files=[]
if hasattr(args, 'city_Incerm') and hasattr(args, 'city_original'):
    for city in args.city_original:
        files.append(f"{args.Increm_root}/{city}_gen.txt")
    for city in args.city_Incerm:
        files.append(f"{args.train_root}/{city}_train.txt")
    
else:
    for city in args.target_city:
        files.append(f"{args.train_root}/{city}_train.txt")
for file_path in files:
    if os.path.exists(file_path):
        shutil.copy(file_path, frozen_test_file)
    else:
        print(f"Warning: {file_path} 不存在，跳过复制")

froze_info_file=log_dir+"/froze_info_file"
os.makedirs(froze_info_file, exist_ok=True)

args.test_root=frozen_test_file

if hasattr(args, 'city_Incerm') and hasattr(args, 'city_original'):
    city_test = args.city_original.copy()  # 复制一个新列表
    city_test.extend(args.city_Incerm)     # 扩展不会影响原始列表
    model = Traj_Model(Traj_Config(n_embd=args.n_embd,
                            n_head=args.n_head,
                            n_layer=args.n_layer,
                            num_experts=args.num_experts+args.add_exp_num,
                            top_k=args.top_k,
                            target_city=city_test)).to(args.device)
    print(f"model_{args.city_Incerm[0]}")
    model.load_state_dict(torch.load(log_dir + f"/model_{args.city_Incerm[0]}.pth",weights_only=True))
    model.eval()
    for city in city_test:
        print(f'{city}')
        print(args.test_root)
        city = [f'{city}']
        test_loader = get_dataloader(args.test_root, city, args.B, args.T,few_shot=False)
        model.eval()
        evaluate(model,
                 args,
                 test_loader=test_loader,
                 log_dir=log_dir,
                 B=args.B,
                 city=city,
                 device=args.device,
                 froze_info_file=froze_info_file)
    
    
else:
    for city in args.target_city:
        model = Traj_Model(Traj_Config(n_embd=args.n_embd,
                                    n_head=args.n_head,
                                    n_layer=args.n_layer,
                                    num_experts=args.num_experts,
                                    top_k=args.top_k,
                                    target_city=args.target_city
                                    )).to(args.device)
        city = [f'{city}']
        print("all")
        print(args.test_root)
        model.load_state_dict(torch.load(log_dir + f"/model_all.pth",weights_only=True))
        model.eval()
        test_loader = get_dataloader(args.test_root, city, args.B, args.T,few_shot=False)
        evaluate(model,
                 args,
                 test_loader=test_loader,
                 log_dir=log_dir,
                 B=args.B,
                 city=city,
                 device=args.device,
                 froze_info_file=froze_info_file)

##----## get to froze ##----##
if hasattr(args, 'city_Incerm') and hasattr(args, 'city_original'):
    city_to_get = args.city_original.copy()  # 复制一个新列表
    city_to_get.extend(args.city_Incerm)
else:
    city_to_get=args.target_city
layer_max_indices = {}
print(city_to_get)
for city in city_to_get:
    path=froze_info_file+f"/gate_city_avg_{city}.txt"
    count=0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Layer"):
                layer_num = int(line.split(":")[0].split()[-1])
                values = eval(line.split(":", 1)[1].strip())
                max_idx = values.index(max(values))
                print(f"{city}:{count}:max_idx{max_idx}")
                count+=1
                if layer_num not in layer_max_indices:
                    layer_max_indices[layer_num] = set()
                layer_max_indices[layer_num].add(max_idx)

# 输出结果
for layer in sorted(layer_max_indices.keys()):
    print(f"{sorted(layer_max_indices[layer])},")

output_txt_path = os.path.join(froze_info_file, "layer_max_indices.txt")

with open(output_txt_path, "w") as f:
    for layer in sorted(layer_max_indices.keys()):
        indices = sorted(layer_max_indices[layer])
        f.write(f"{indices},\n")