import os
import ast
import numpy as np
import torch
from torch.nn import functional as F
torch.cuda.empty_cache()  # Clear the cache
from dataloader_detla_lt import TrajDataset_Incremental,get_dataloader
import random
import argparse
from torch.utils.data import DataLoader
from utils_delta import evaluate,train_stop
from base_model import Traj_Config,Traj_Model 
from updated_model import Traj_Incremental_Config,Traj_Model_Incremental

parser = argparse.ArgumentParser(description="traj_MOE_MODEL")

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

parser.add_argument('--device', default='cuda', help='Device for Attack')

parser.add_argument('--teacher_model', 
                    default="./base_model/['WashingtonDC', 'Seattle', 'Chicago']/6_512", 
                    help='Iteacher_model')

parser.add_argument('--Increm_root', default="./traj_data_delta_tl/gen_0510/20_final/NewYork_['WashingtonDC', 'Atlanta', 'LosAngeles']_log_1.0", help='Increm_root')
parser.add_argument('--city_Incerm',  nargs='+', default=['NewYork'])

parser.add_argument('--add_exp_num',  default=1)    

parser.add_argument('--idx_add_exp',  nargs='+', default=[0,1,2,3,4,5])

parser.add_argument("--epoch", type=int, default=30)

parser.add_argument("--B", type=int, default=128)

parser.add_argument("--epoch", type=int, default=1.2e-4)

parser.add_argument('--experts_froze', type=ast.literal_eval, default=[[0, 1, 2],
                                                                       [0, 1, 3],
                                                                       [1, 2, 3],
                                                                       [1, 2, 3],
                                                                       [1, 3],
                                                                       [0]])

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
    将字符串值转换为相应的 Python 对象（如列表、数字等）
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

def update_args_with_params(args, params):
    """
    将 params 中的参数加入 args 中。
    若 args 中已有该参数（通过命令行或已有默认值），则不覆盖。
    """
    for key, value in params.items():
        if hasattr(args, key):
            # args 中已有该参数，不覆盖
            continue
        else:
            # 自动类型转换（数字、布尔等）
            converted_value = convert_value(value)
            setattr(args, key, converted_value)

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)                   
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parser.parse_args()
log_dir = args.teacher_model
log_settings_path=log_dir+"/log_settings.txt"

if "updated_model" in args.teacher_model:
    params = load_log_settings(log_settings_path)
    update_args_with_params(args, params)
    args.num_experts+=args.add_exp_num
    args.add_exp_num=1
    args.experts_froze[0]=list(range(len(args.city_original)+len(args.city_Incerm)+1))
    args.epoch=30
    target_city=args.target_city.copy()
    target_city.extend(params.city_Incerm[0])
    args.target_city=target_city
    city_origi_model=params.city_Incerm[0]
    city_original=args.city_original.copy()

    city_original.extend(params.city_Incerm[0])

    args.city_original=city_original
    city_test=city_original.copy()
    city_test.extend(args.city_Incerm)
    args.city_test=city_test
    args.city=args.city_original
else:
    params = load_log_settings(log_settings_path)
    update_args_with_params(args, params)
    args.city_original=args.city
    city_test=args.city.copy()
    city_test.extend(args.city_Incerm)
    args.city_test=city_test
    args.experts_froze[0]=list(range(len(args.city_original)+len(args.city_Incerm)+1))
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

EPK=None
stage_train=None
froze_bottom_all=True
stage_unfroze=True
froze_top1=True

set_random_seed(args.seed)

log_dir_new_model=f"./updated_model/{args.city}_Increm{args.city_Incerm}_{args.n_layer}_{args.n_embd}/experts_froze{args.experts_froze}"

os.makedirs(log_dir_new_model, exist_ok=True)
log_settings = os.path.join(log_dir_new_model, f"log_settings.txt")

with open(log_settings, "w") as f:
    for arg, value in vars(args).items():
        f.write(f"{arg}: {value}\n")

cities=args.city.copy()
cities.extend(args.city_Incerm)
args.city=cities
args.target_city=cities
print(args.target_city)

print(args.city_original)

print(f"city_origi_model:{city_origi_model}")
model_origin = Traj_Model(Traj_Config(n_embd=args.n_embd,
                                    n_head=args.n_head,
                                    n_layer=args.n_layer,
                                    num_experts=args.num_experts,
                                    top_k=args.top_k,
                                    target_city=args.city_original)).to(args.device)

model_origin.load_state_dict(torch.load(log_dir + f"/model_{city_origi_model}.pth",weights_only=True))
model_origin.to(args.device)

teacher_model=Traj_Model(Traj_Config(n_embd=args.n_embd,
                                    n_head=args.n_head,
                                    n_layer=args.n_layer,
                                    num_experts=args.num_experts,
                                    top_k=args.top_k,
                                    target_city=args.city_original)).to(args.device)
teacher_model.load_state_dict(torch.load(log_dir + f"/model_{city_origi_model}.pth",weights_only=True))
teacher_model.to(args.device)

model_new =Traj_Model_Incremental(config=Traj_Incremental_Config(
                                                                n_embd=args.n_embd,
                                                                n_head=args.n_head,
                                                                n_layer=args.n_layer,
                                                                num_experts=args.num_experts,
                                                                top_k=args.top_k,
                                                                add_exp_num=args.add_exp_num,
                                                                idx_add_exp=args.idx_add_exp,
                                                                experts_froze=args.experts_froze,
                                                                vocab_embed_froze=args.vocab_embed_froze,
                                                                pos_tim_embed_froze=args.pos_tim_embed_froze,
                                                                different_lr=args.different_lr,
                                                                router_model=args.router_model,
                                                                add_block=args.add_block,
                                                                add_block_place=args.add_block_place,
                                                                city_original=args.city_original,
                                                                city_Incerm=args.city_Incerm,
                                                                city_target=args.target_city
                                                                ),
                                  model=model_origin).to(args.device)

train_dataset = TrajDataset_Incremental(args.train_root, args.Increm_root,args.city_original,args.city_Incerm, args.B, args.T,args.few_shot)
train_loader = DataLoader(train_dataset, batch_size=args.B, shuffle=False)


valid_step_interval = len(train_dataset)//args.B
val_loader = []
for city in args.target_city:
    city = [f'{city}']
    val_loader_city = get_dataloader(args.val_root, city, args.B, args.T,few_shot=False)
    val_loader.append(val_loader_city)

train_stop(model_new,
           args,
           train_loader=train_loader,
           valid_loaders=val_loader,
           log_dir=log_dir_new_model,
           lr=args.lr,
           epoch=args.epoch,
           valid_step_interval=valid_step_interval,
           device = args.device,
           citys = args.target_city,
           patience=10,
           Increm=True,
           distill=True,
           teacher_model=teacher_model,
           stage_unfroze=stage_unfroze)

model_new.load_state_dict(torch.load(log_dir_new_model + f"/model_{args.city_Incerm[0]}.pth",weights_only=True))

for city in args.target_city:
    city = [f'{city}']
    test_loader = get_dataloader(args.test_root, city, args.B, args.T,few_shot=False)
    evaluate(model_new,config=args,test_loader=test_loader,log_dir=log_dir_new_model,B=args.B,city=city,device=args.device)