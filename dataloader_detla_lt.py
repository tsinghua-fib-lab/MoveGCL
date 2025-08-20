import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

import random

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
        # print(shards)
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
            random.shuffle(self.data_list)
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
                for part in parts:
                    if part:  
                        location, day, time, delta_time, quantized_distance = part.split(',')
                        day = int(day)
                        time = int(time)
                        delta_time=int(delta_time)
                        quantized_distance= int(quantized_distance)
                        rg=int(rg)
                        entropy=int(entropy)
                        traj.append([int(location) + 2, time,delta_time,quantized_distance,rg,entropy])

                traj.append([int(1), int(0), int(0), int(0), int(0), int(0)])
                for _ in range(self.T+1 - len(traj)):
                    traj.append([int(0), int(0), int(0), int(0), int(0), int(0)])
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
        return x, y, ts_his,delta_t_his,delta_dis_his,rg,entropy,file


def get_dataloader(data_root, split, B, T,few_shot,LPR=False):
    dataset = TrajDataset(data_root, split, B,T,few_shot,LPR=LPR)
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=B, shuffle=False)
    return dataloader

def get_dataloader_LRP(data_root, split, B, T,few_shot,LPR=False):
    dataset = TrajDataset(data_root, split, B,T,few_shot,LPR=LPR)
    dataset_len=len(dataset)
    print(dataset_len)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=False)
    return dataloader,dataset_len



class TrajDataset_Incremental(Dataset):
    def __init__(self, train_root,Increm_root, split_city_original,split_city_Incerm, B,T,few_shot,LPR=False):
        self.B = B
        self.T = T
        self.split_city_original = split_city_original
        self.split_city_Incerm = split_city_Incerm
        self.few_shot = few_shot
        # load the shards
        shards = os.listdir(Increm_root)
        # sprint(split_city_original)
        shards = [s for s in shards if any(x in s for x in split_city_original)]
        shards = [os.path.join(Increm_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split_city_original}"
        print(f"found {len(shards)} shards for split {split_city_original}")

        shards1 = os.listdir(train_root)
        shards1 = [s for s in shards1 if any(x in s for x in split_city_Incerm)]
        shards1 = [os.path.join(train_root, s) for s in shards1]
        assert len(shards1) > 0, f"no shards found for split {split_city_Incerm}"
        print(f"found {len(shards1)} shards for split {split_city_Incerm}")
        shards.extend(shards1)
        self.shards = shards
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
        # print(shard_indices)

        remaining_batches = {shard: len(batches[shard]) for shard in self.shards}
        print(remaining_batches)

        # print(shard_indices)
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
                    batch = batches[shard][shard_indices[shard]]
                    self.data_list.append(batch)

                    shard_indices[shard] += 1
                    remaining_batches[shard] -= 1
            random.shuffle(self.data_list)
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
                for part in parts:
                    if part:  
                        # print(part)
                        try:
                            location, day, time, delta_time, quantized_distance = part.split(',')
                        except:
                            print(line)
                        day = int(day)
                        time = int(time)
                        delta_time=int(delta_time)
                        quantized_distance= int(quantized_distance)
                        rg=int(rg)
                        entropy=int(entropy)
                        traj.append([int(location) + 2, time,delta_time,quantized_distance,rg,entropy])

                traj.append([int(1), int(0), int(0), int(0), int(0), int(0)])
                for _ in range(self.T+1 - len(traj)):
                    traj.append([int(0), int(0), int(0), int(0), int(0), int(0)])
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
        return x, y, ts_his,delta_t_his,delta_dis_his,rg,entropy,file




class TrajDataset_Incremental_LRP(Dataset):
    def __init__(self,
                 n_layer,
                 num_experts,
                 train_root,
                 Increm_root, 
                 split_city_original,
                 split_city_Incerm, 
                 B,T,few_shot,loaded_gate_top1,loaded_gate_top2):
        
        self.B = B
        self.T = T
        self.split_city_original = split_city_original
        self.split_city_Incerm = split_city_Incerm
        self.few_shot = few_shot
        # load the shards
        shards = os.listdir(Increm_root)
        # sprint(split_city_original)
        shards = [s for s in shards if any(x in s for x in split_city_original)]
        shards = [os.path.join(Increm_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split_city_original}"
        print(f"found {len(shards)} shards for split {split_city_original}")

        shards1 = os.listdir(train_root)
        shards1 = [s for s in shards1 if any(x in s for x in split_city_Incerm)]
        shards1 = [os.path.join(train_root, s) for s in shards1]
        assert len(shards1) > 0, f"no shards found for split {split_city_Incerm}"
        print(f"found {len(shards1)} shards for split {split_city_Incerm}")
        shards.extend(shards1)
        self.shards = shards
        

        shard_to_city={}
        for idx,shard in enumerate(shards):
            if idx<len(split_city_original):
                shard_to_city[shard]=split_city_original[idx]
            else:
                shard_to_city[shard]=split_city_Incerm[idx-len(split_city_original)]

        self.data_city = defaultdict()
        for shard in self.shards:
            self.data_city[shard] = self.load_traj(shard)
        
        self.data = []
        self.gate_top1_data = []
        self.gate_top2_data = []

        # train_len=int(len(self.data_city[shard])//self.B)*self.B
        batches = {shard: [self.data_city[shard][i:i + self.B] 
                           for i in range(0, int(len(self.data_city[shard])//self.B)*self.B, self.B)] 
                   for shard in self.shards}
        
        
        batches_gate_top1={}
        for shard,city in list(shard_to_city.items()):
            batches_gate_top1[shard]=[]
            for i in range(0, int(len(self.data_city[shard])//self.B)*self.B, self.B):
                if city in split_city_original:
                    # print(1)
                    batches_gate_top1[shard].append(loaded_gate_top1[city][:,i:i + self.B,:])
                else:
                    # print(2)
                    batches_gate_top1[shard].append(torch.full((n_layer, B, T), fill_value=-1, dtype=torch.float32))
        
        batches_gate_top2={}
        for shard,city in list(shard_to_city.items()):
            batches_gate_top2[shard]=[]
            for i in range(0, int(len(self.data_city[shard])//self.B)*self.B, self.B):
                if city in split_city_original:
                    batches_gate_top2[shard].append(loaded_gate_top2[city][:,i:i + self.B,:])
                else:
                    batches_gate_top2[shard].append(torch.full((n_layer, B, T), fill_value=-1, dtype=torch.float32))


        shard_indices = {shard: 0 for shard in self.shards}
        remaining_batches = {shard: len(batches[shard]) for shard in self.shards}
        print(remaining_batches)

        self.data_list=[]
        self.gate_data_top1_list=[]
        self.gate_data_top2_list=[]
        for shard in self.shards:
            # print(shard)
            # print(len(batches[shard]))
            for i in range(len(batches[shard])):
                # print(i)
                batch = batches[shard][shard_indices[shard]]
                batch_gate_top1=batches_gate_top1[shard][shard_indices[shard]]
                batch_gate_top1 = batch_gate_top1.permute(1, 0, 2)
                batch_gate_top2=batches_gate_top2[shard][shard_indices[shard]]
                batch_gate_top2 = batch_gate_top2.permute(1, 0, 2)
                self.data_list.append(batch)
                self.gate_data_top1_list.append(batch_gate_top1)
                self.gate_data_top2_list.append(batch_gate_top2)

                shard_indices[shard] += 1
                remaining_batches[shard] -= 1
        # 创建一个索引列表
        indices = list(range(len(self.data_list)))

        # 随机打乱索引列表
        random.shuffle(indices)

        # 根据打乱后的索引重新排列数据
        self.data_list = [self.data_list[i] for i in indices]
        self.gate_data_top1_list = [self.gate_data_top1_list[i] for i in indices]
        self.gate_data_top2_list = [self.gate_data_top2_list[i] for i in indices]

        for trajs in self.data_list:
            self.data.extend(trajs)
        
        for gates in self.gate_data_top1_list:
            self.gate_top1_data.extend(gates)

        for gates in self.gate_data_top2_list:
            self.gate_top2_data.extend(gates)

                
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
                for part in parts:
                    if part:  
                        location, day, time, delta_time, quantized_distance = part.split(',')
                        day = int(day)
                        time = int(time)
                        delta_time=int(delta_time)
                        quantized_distance= int(quantized_distance)
                        rg=int(rg)
                        entropy=int(entropy)
                        traj.append([int(location) + 2, time,delta_time,quantized_distance,rg,entropy])

                traj.append([int(1), int(0), int(0), int(0), int(0), int(0)])
                for _ in range(self.T+1 - len(traj)):
                    traj.append([int(0), int(0), int(0), int(0), int(0), int(0)])
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
        gate_top1_choice=self.gate_top1_data[idx]
        gate_top2_choice=self.gate_top2_data[idx]
        # print(gate_choice)        
        x = traj[:-1, 0]
        y = traj[1:, 0]
        ts_his = traj[:-1, 1]
        delta_t_his=traj[:-1, 2]
        delta_dis_his=traj[:-1,3]
        rg=traj[:-1,4]
        entropy=traj[:-1,5]
        return x, y, ts_his,delta_t_his,delta_dis_his,rg,entropy,file,gate_top1_choice,gate_top2_choice


class TrajDataset_distill(Dataset):
    def __init__(self,
                 train_root,
                 Increm_root, 
                 split_city_original,
                 split_city_Incerm, 
                 B,
                 T,
                 few_shot):
        
        self.B = B
        self.T = T
        self.split_city_original = split_city_original
        self.split_city_Incerm = split_city_Incerm
        self.few_shot = few_shot
        self.vocab_len_list={
            "Atlanta":1177,
            "WashingtonDC":1363,
            "NewYork":4990,
            "Seattle":1048,
            "LosAngeles":6200,
            "Chicago":3599

        }
        # load the shards
        shards = os.listdir(Increm_root)
        # sprint(split_city_original)
        shards = [s for s in shards if any(x in s for x in split_city_original)]
        shards = [os.path.join(Increm_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split {split_city_original}"
        print(f"found {len(shards)} shards for split {split_city_original}")

        shards1 = os.listdir(train_root)
        shards1 = [s for s in shards1 if any(x in s for x in split_city_Incerm)]
        shards1 = [os.path.join(train_root, s) for s in shards1]
        assert len(shards1) > 0, f"no shards found for split {split_city_Incerm}"
        print(f"found {len(shards1)} shards for split {split_city_Incerm}")
        shards.extend(shards1)
        self.shards = shards

        shards_distill=os.listdir(Increm_root+"/distill")
        # sprint(split_city_original)
        shards_distill = [s for s in shards_distill if any(x in s for x in split_city_original)]
        shards_distill = [os.path.join(Increm_root+"/distill", s) for s in shards_distill]
        assert len(shards_distill) > 0, f"no shards_distill found for split {split_city_original}"
        print(f"found {len(shards_distill)} shards_distill for split {split_city_original}")
        self.shards_distill = shards_distill
        

        shard_to_city={}
        city_to_shard={}
        for idx,shard in enumerate(shards):
            if idx<len(split_city_original):
                shard_to_city[shard]=split_city_original[idx]
                city_to_shard[split_city_original[idx]]=shard
            else:
                shard_to_city[shard]=split_city_Incerm[idx-len(split_city_original)]
                city_to_shard[split_city_Incerm[idx-len(split_city_original)]]=shard
        
        shards_distill_to_city={}
        for idx,shard in enumerate(shards_distill):
            if idx<len(split_city_original):
                shards_distill_to_city[shard]=split_city_original[idx]
            else:
                shards_distill_to_city[shard]=split_city_Incerm[idx-len(split_city_original)]

        shards_distill_to_shard={}
        for shard_distill,city in list(shards_distill_to_city.items()):
            shards_distill_to_shard[shard_distill]=city_to_shard[city]

        self.data_city = defaultdict()
        for shard in self.shards:
            self.data_city[shard] = self.load_traj(shard)

        self.data_distill = defaultdict()
        for shard_distill in self.shards_distill:
            # print(shard)
            self.data_distill[shards_distill_to_shard[shard_distill]] = torch.load(shard_distill,weights_only=True)

        
        self.data = []
      
        batches = {shard: [self.data_city[shard][i:i + self.B] 
                           for i in range(0, int(len(self.data_city[shard])//self.B)*self.B, self.B)] 
                   for shard in self.shards}
        
        
        batches_distill={}
        for shard,city in list(shard_to_city.items()):
            # print(shard)
            # print(city)
            batches_distill[shard]=[]
            # print(batches_distill)
            for i in range(0, int(len(self.data_city[shard])//self.B)*self.B, self.B):
                if city in split_city_original:
                    batches_distill[shard].append(self.data_distill[shard][i:i + self.B,:,:])
                else:
                    batches_distill[shard].append(torch.full((B, T,self.vocab_len_list[city]), fill_value=-1, dtype=torch.float32))
        


        shard_indices = {shard: 0 for shard in self.shards}
        remaining_batches = {shard: len(batches[shard]) for shard in self.shards}
        print(remaining_batches)

        self.data_list=[]
        self.data_distill_list=[]
        self.distill_data=[]
        # self.gate_data_top2_list=[]
        for shard in self.shards:
            # print(shard)
            # print(len(batches[shard]))
            for i in range(len(batches[shard])):
                # print(i)
                batch = batches[shard][shard_indices[shard]]
                batch_gate_distill=batches_distill[shard][shard_indices[shard]]
                # batch_gate_top1 = batch_gate_top1.permute(1, 0, 2)
                self.data_list.append(batch)
                self.data_distill_list.append(batch_gate_distill)

                shard_indices[shard] += 1
                remaining_batches[shard] -= 1
        # 创建一个索引列表
        indices = list(range(len(self.data_list)))

        # 随机打乱索引列表
        random.shuffle(indices)

        # 根据打乱后的索引重新排列数据
        self.data_list = [self.data_list[i] for i in indices]
        self.data_distill_list = [self.data_distill_list[i] for i in indices]

        for trajs in self.data_list:
            self.data.extend(trajs)
        
        for gates in self.data_distill_list:
            self.distill_data.extend(gates)

                
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
                for part in parts:
                    if part:  
                        location, day, time, delta_time, quantized_distance = part.split(',')
                        day = int(day)
                        time = int(time)
                        delta_time=int(delta_time)
                        quantized_distance= int(quantized_distance)
                        rg=int(rg)
                        entropy=int(entropy)
                        traj.append([int(location) + 2, time,delta_time,quantized_distance,rg,entropy])

                traj.append([int(1), int(0), int(0), int(0), int(0), int(0)])
                for _ in range(self.T+1 - len(traj)):
                    traj.append([int(0), int(0), int(0), int(0), int(0), int(0)])
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
        distill_data=self.distill_data[idx]
        # print(gate_choice)        
        x = traj[:-1, 0]
        y = traj[1:, 0]
        ts_his = traj[:-1, 1]
        delta_t_his=traj[:-1, 2]
        delta_dis_his=traj[:-1,3]
        rg=traj[:-1,4]
        entropy=traj[:-1,5]
        return x, y, ts_his,delta_t_his,delta_dis_his,rg,entropy,file,distill_data




# 示例用法
# if __name__ == "__main__":
#     data_root = '../traj_dataset/mini/val'
#     city = ['nanchang','lasa']  # 或 'valid', 'test'
#     B = 16
#     T = 144
#     dataloader = get_dataloader(data_root, city, B, T,few_shot=1.0)
#     for batch_no, train_batch in enumerate(dataloader, start=1):
#         print(train_batch[0].size())
#         print(train_batch[1].size())
#         print(train_batch[2].size())
#         print(train_batch[3][0])


        