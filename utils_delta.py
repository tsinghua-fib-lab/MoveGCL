import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import os
from draw_loss_cur import draw_train_info
import math
import torch.nn.functional as F

def train_stop(model,
               config, 
               train_loader, 
               valid_loaders, 
               log_dir, 
               lr, 
               epoch, 
               valid_step_interval, 
               device, 
               citys, 
               patience=10,
               LPR_train=False,
               Increm=False,
               distill=False,
               teacher_model=None,
               stage_unfroze=None,
               EPK=None):
    
    log_file_train = os.path.join(log_dir, f"log_train.txt")
    with open(log_file_train, "w") as f:  # open for writing to clear the file
        pass

    if Increm==True:
        if stage_unfroze:
            optimizer = model.configure_optimizers(weight_decay=0.1, 
                                                learning_rate=lr,
                                                epoch_part=0,
                                                LPR_train=LPR_train,
                                                stage_unfroze=stage_unfroze)
            stage1_end = int(0.3 * epoch)
            stage2_end = int(0.6 * epoch)
        
            class StagewiseScheduler:
                def __init__(self, optimizer, stages):
                    self.optimizer = optimizer
                    self.stages = stages
                    self.current_stage = 0
                    self.current_epoch = 0
                    self.base_lr = lr
                    
                def step(self):
                    self.current_epoch += 1
                    if self.current_stage < len(self.stages)-1 and self.current_epoch >= self.stages[self.current_stage]['end_epoch']:
                        self.current_stage += 1
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.base_lr
                     
                    current_stage_config = self.stages[self.current_stage]
                    if ((self.current_epoch - current_stage_config['start_epoch']) % current_stage_config['step_size'] == 0) and (self.current_epoch - current_stage_config['start_epoch'])!=0:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= current_stage_config['gamma']
            

            stages = [
                {'start_epoch': 0, 'end_epoch': stage1_end, 'step_size': 3, 'gamma': 0.4}, 
                {'start_epoch': stage1_end, 'end_epoch': stage2_end, 'step_size': 3, 'gamma': 0.4},
                {'start_epoch': stage2_end, 'end_epoch': epoch, 'step_size': 3, 'gamma': 0.4}
            ]
            
            lr_scheduler = StagewiseScheduler(optimizer, stages)
           
        else:
            optimizer = model.configure_optimizers(weight_decay=0.1, 
                                               learning_rate=lr,
                                               epoch_part=0,
                                               LPR_train=LPR_train)
            p1 = int(0.2 * epoch)
            p2 = int(0.4 * epoch)
            p3 = int(0.6 * epoch)
            p4 = int(0.8 * epoch)

            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[p1, p2, p3, p4], gamma=0.4
            )

    else:
        optimizer = model.configure_optimizers(weight_decay=0.1, 
                                               learning_rate=lr,
                                               LPR_train=LPR_train)
        
        p1 = int(0.2 * epoch)
        p2 = int(0.4 * epoch)
        p3 = int(0.6 * epoch)
        p4 = int(0.8 * epoch)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2, p3, p4], gamma=0.4
        )
    
    if distill:
        teacher_model.eval()


    best_valid_loss_dict = {}
    patience_counter_dict = {}

    for city in citys:
        best_valid_loss_dict[f'{city}'] = 0
        patience_counter_dict[f'{city}'] = 0
        log_file_val = os.path.join(log_dir, f"log_val_{city}.txt")
        with open(log_file_val, "w") as f:
            pass
    
    log_file_val = os.path.join(log_dir, f"log_val_all.txt")
    with open(log_file_val, "w") as f:
        pass

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    best_loss_all=1e10

    for epoch_no in range(epoch):
        for param_group in optimizer.param_groups:
            param_group['epoch_part'] = epoch_no

        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            accumulation_steps = 16
            accumulated_samples = 0
            accumulated_loss = 0.0
            for batch_no, train_batch in enumerate(it, start=1):
                current_lr = optimizer.param_groups[0]['lr']
                # print(current_lr)
                if valid_loaders is not None and (batch_no + 1) % valid_step_interval == 0:
                    model.eval()
                    average_loss_all = 0
                    count_all=0
                    size = 0
                    acc1 = 0
                    acc3 = 0
                    acc5 = 0
                    
                    acc1_city={}
                    for city in config.target_city:
                        acc1_city[city]=0
                    
                    acc3_city={}
                    for city in config.target_city:
                        acc3_city[city]=0
                    
                    acc5_city={}
                    for city in config.target_city:
                        acc5_city[city]=0

                    size_city={}
                    for city in config.target_city:
                        size_city[city]=0
                    
                    with torch.no_grad():
                        for valid_loader in valid_loaders:
                            avg_loss_valid = 0
                            with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                                for batch_no_val, valid_batch in enumerate(it, start=1):
                                    x_val = valid_batch[0]
                                    y_val = valid_batch[1]
                                    ts = valid_batch[2]
                                    delta_ts=valid_batch[3]
                                    delta_dis_his = valid_batch[4]
                                    rg = valid_batch[5]
                                    entropy= valid_batch[6]

                                    val_city = valid_batch[7][0]
                                    vocab = np.load(f'./location_feature/vocab_{val_city}.npy')
                                    vocab = np.pad(vocab, ((2, 0), (0, 0)), mode='constant', constant_values=0)
                                    vocab = torch.from_numpy(vocab)
                                    vocab = vocab.to(torch.float32)
                                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                                        output,gate_all = model(x_val, 
                                                                ts, 
                                                                y_val, 
                                                                delta_ts,
                                                                delta_dis_his,
                                                                rg,
                                                                entropy,
                                                                vocab, 
                                                                val_city,
                                                                device)
                                    loss = output['loss']
                                    average_loss_all+= loss.item()
                                    count=batch_no_val
                                    avg_loss_valid += loss.item()
                                    loss_avg_valid = avg_loss_valid / batch_no_val

                                    it.set_postfix(
                                        ordered_dict={
                                            "valid_avg_loss": loss_avg_valid,
                                            "epoch": epoch_no,
                                        },
                                        refresh=False,
                                    )

                                    pred = output['logits']#[B T vocab_size]
                                    pred[:,:,0] = float('-inf')   
                                    y_val = y_val.to(device)

                                    for b in range(len(y_val)):
                                        if b >= pred.size(0):
                                            break
                                        _, pred_indices = torch.topk(pred[b], 100)
                                        valid_mask = y_val[b] > 0
                                        valid_y_val = y_val[b][valid_mask]
                                        valid_pred_indices = pred_indices[valid_mask]
                            
                                        valid_y_val_expanded = valid_y_val.unsqueeze(1) 
                                        l= valid_y_val_expanded.size(0)
                                        size +=l
                                        size_city[val_city]+=l
                            
                                        a1 = torch.sum(valid_pred_indices[:, 0:1] == valid_y_val_expanded).item()
                                        a3 = torch.sum(valid_pred_indices[:, 0:3] == valid_y_val_expanded).item()
                            
                                        a5 = torch.sum(valid_pred_indices[:,0:5] == valid_y_val_expanded).item()
                                        acc1 += a1
                                        acc1_city[val_city] += a1
                                        acc3 += a3
                                        acc3_city[val_city] += a3
                                        acc5 += a5
                                        acc5_city[val_city] += a5

                                count_all+=count
                                log_file_val = os.path.join(log_dir, f"log_val_{val_city}.txt")
                                with open(log_file_val, "a") as f:
                                    f.write(f"{epoch_no}\t{batch_no}\t val \t{loss_avg_valid:.6f}\t acc1 \t{acc1_city[val_city]/size_city[val_city]}\t acc3 \t{acc3_city[val_city]/size_city[val_city]}\t acc5 \t{acc5_city[val_city]/size_city[val_city]}\t lr \t{current_lr}\t \n")

                                if best_valid_loss_dict[f'{val_city}'] <acc1_city[val_city]/size_city[val_city]:
                                        output_path = log_dir + f"/model_{val_city}.pth"
                                        torch.save(model.state_dict(), output_path)
                                        best_valid_loss_dict[f'{val_city}'] = acc1_city[val_city]/size_city[val_city]
                                        patience_counter_dict[f'{val_city}'] = 0 
                                        print(
                                            "\n best loss is updated to ",
                                            loss_avg_valid,
                                            "at",
                                            epoch_no, val_city
                                        )
                                else:
                                    patience_counter_dict[f'{val_city}'] += 1
                                    if all(value > patience for value in patience_counter_dict.values()):
                                        print(f"\n Early stopping triggered for {val_city} at epoch {epoch_no}")

                                    
                    if best_loss_all > average_loss_all/count_all:
                        output_path = log_dir + f"/model_all.pth"
                        torch.save(model.state_dict(), output_path)
                        best_loss_all = average_loss_all/count_all

                    log_file_val = os.path.join(log_dir, f"log_val_all.txt")
                    with open(log_file_val, "a") as f:
                        f.write(f"{epoch_no}\t{batch_no}\t val \t{average_loss_all/count_all:.6f}\t acc1 \t{acc1/size}\t acc3 \t{acc3/size}\t acc5 \t{acc5/size}\t lr \t{current_lr}\t \n")

                    for city in citys:
                        draw_train_info(log_dir,city,valid_step_interval*4)

                optimizer.zero_grad()
                x_train = train_batch[0]
                y_train = train_batch[1]
                ts = train_batch[2]
                delta_ts = train_batch[3]
                delta_dis_his = train_batch[4]
                rg = train_batch[5]
                entropy= train_batch[6]
                train_city = train_batch[7][0]
               
                vocab = np.load(f'./location_feature/vocab_{train_city}.npy')
                vocab = np.pad(vocab, ((2, 0), (0, 0)), mode='constant', constant_values=0)
                vocab = torch.from_numpy(vocab)
                vocab = vocab.to(torch.float32)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    output,gate_all = model(x_train, 
                                            ts,
                                            y_train, 
                                            delta_ts,
                                            delta_dis_his,
                                            rg,
                                            entropy,
                                            vocab, 
                                            train_city,
                                            device)
                loss = output['loss']

                if distill and train_city in config.city_original:
                    mask = (x_train != 0)
                    mask = mask.to(device)
                    with torch.no_grad():
                        output_distill,_ = teacher_model(x_train, 
                                                ts,
                                                y_train, 
                                                delta_ts,
                                                delta_dis_his,
                                                rg,
                                                entropy,
                                                vocab, 
                                                train_city,
                                                device)
                    teacher_logits=output_distill['logits']
                    student_logits= output['logits']
                    # Step 1: logits for teacher
                    temperature = 2.0
                    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

                    # Step 2: log_softmax for student
                    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

                    # Step 3: Knowledge Distillation Loss
                    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none') * (temperature ** 2) # [B, T, vocab_size]

                    masked_kl = kl_loss * mask.unsqueeze(-1)
                    loss = masked_kl.sum() / mask.sum()

                
                current_batch_size = x_train.size(0)
                accumulated_loss += loss.item() * current_batch_size
                accumulated_samples += current_batch_size
                loss.backward()
                

                if accumulated_samples >= accumulation_steps:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    

                    avg_loss = accumulated_loss / accumulated_samples
                    with open(log_file_train, "a") as f:
                        f.write(f"{epoch_no}\t{batch_no}\t train \t{avg_loss:.6f}\n")
                    

                    accumulated_loss = 0.0
                    accumulated_samples = 0
                

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss,
                        "epoch": epoch_no,
                        "lr": current_lr,
                    },
                    refresh=False,
                )
            if accumulated_samples > 0:
                optimizer.step()
                optimizer.zero_grad()
                avg_loss = accumulated_loss / accumulated_samples
                with open(log_file_train, "a") as f:
                    f.write(f"{epoch_no}\t{batch_no}\t train \t{avg_loss:.6f}\n")

        lr_scheduler.step()



def evaluate(model,config,test_loader,log_dir,B,city,device):
    log_file_test = os.path.join(log_dir, f"log_{city}_test.txt")
    with open(log_file_test, "w") as f:
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
                # print(f"x_test:{x_test.size()}")
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
                    # print(gate_city)
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
    gate_file = os.path.join(log_dir, f"gate_city_avg_{city[0]}.txt")
    with open(gate_file, "w") as f:
        for city_name, gate_dict in gate_city_avg.items():
            f.write(f"City: {city_name}\n")
            for idx, gate_tensor in gate_dict.items():
                gate_list = gate_tensor.cpu().tolist()
                f.write(f"Layer {idx}: {gate_list}\n")

    gate_file = os.path.join(log_dir, f"gate_city_{city[0]}.txt")
    with open(gate_file, "w") as f:
        for city_name, gate_dict in gate_city.items():
            f.write(f"City: {city_name}\n")
            for idx, gate_tensor in gate_dict.items():
                gate_list = gate_tensor.cpu().tolist()
                f.write(f"Layer {idx}: {gate_list}\n")