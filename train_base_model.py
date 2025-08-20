import os
import numpy as np
import torch
from torch.nn import functional as F
torch.cuda.empty_cache()  # Clear the cache
from dataloader_detla_lt import TrajDataset,get_dataloader
import random
import argparse
from torch.utils.data import DataLoader
from utils_delta import evaluate,train_stop
from base_model import Traj_Config,Traj_Model 

def set_random_seed(seed: int):
    random.seed(seed)                      
    np.random.seed(seed)                    
    torch.manual_seed(seed)                  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         
        torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

parser = argparse.ArgumentParser(description="traj_MOE_MODEL")

parser.add_argument('--device', default='cuda', help='Device for Attack')
parser.add_argument("--seed", type=int, default=520)
parser.add_argument("--n_embd", type=int, default=512)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--n_layer", type=int, default=6)
parser.add_argument("--num_experts", type=int, default=4)
parser.add_argument("--top_k", type=int, default=2)

#data load
parser.add_argument("--B", type=int, default=16,help='batch size')
parser.add_argument("--T", type=int, default=144,help='max length 48*3days')
parser.add_argument("--city", nargs='+', default=  ['WashingtonDC', 'Seattle', 'Chicago'])
# ['Atlanta', 'WashingtonDC', 'NewYork', 'Seattle', 'LosAngeles', 'Chicago']
parser.add_argument("--train_root", type=str, default='./traj_data/train')
parser.add_argument("--val_root", type=str, default='./traj_data/val')
parser.add_argument("--test_root", type=str, default='./traj_data/test')
parser.add_argument("--few_shot", type=float, default=1.0)
#train
parser.add_argument("--epoch", type=int, default=30)
parser.add_argument("--lr", type=float, default=1.2e-5,help='learning rate')

args = parser.parse_args()
args.target_city=args.city
print(args)
set_random_seed(args.seed)
log_dir = f"./base_model/{args.city}/{args.n_layer}_{args.n_embd}"

os.makedirs(log_dir, exist_ok=True)
log_settings = os.path.join(log_dir, f"log_settings.txt")
with open(log_settings, "w") as f:
    for arg, value in vars(args).items():
        f.write(f"{arg}: {value}\n")


model = Traj_Model(Traj_Config(n_embd=args.n_embd,
                               n_head=args.n_head,
                               n_layer=args.n_layer,
                               num_experts=args.num_experts,
                               top_k=args.top_k,
                               target_city=args.city
                               )).to(args.device)


train_dataset = TrajDataset(args.train_root, args.city, args.B, args.T,args.few_shot)
train_loader = DataLoader(train_dataset, batch_size=args.B, shuffle=False)


valid_step_interval = len(train_dataset)//args.B//4

val_loader = []
for city in args.target_city:
    city = [f'{city}']
    val_loader_city = get_dataloader(args.val_root, city, args.B, args.T,few_shot=False)
    val_loader.append(val_loader_city)

train_stop(model,args,train_loader=train_loader,valid_loaders=val_loader,log_dir=log_dir,lr=args.lr,epoch=args.epoch,valid_step_interval=valid_step_interval,device = args.device,citys = args.target_city,patience=10)


for city in args.target_city:
    model.load_state_dict(torch.load(log_dir + f"/model_{city}.pth",weights_only=True))
    city = [f'{city}']
    test_loader = get_dataloader(args.test_root, city, args.B, args.T,few_shot=False)
    evaluate(model,args,test_loader=test_loader,log_dir=log_dir,B=args.B,city=city,device=args.device)
