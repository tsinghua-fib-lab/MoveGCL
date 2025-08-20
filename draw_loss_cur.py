import matplotlib.pyplot as plt

def draw_train_info(log_file,city,valid_step_interval):
    log_file_val = log_file+f"/log_val_{city}.txt"
    batch_nos = []
    loss_avgs = []
    
    with open(log_file_val, "r") as f:
        for idx,line in enumerate(f):
            parts = line.strip().split("\t")
            epoch_no, batch_no, _, loss_avg_valid = parts[0], parts[1], parts[2], parts[3]
            batch_nos.append(int(batch_no)+1+int(epoch_no)*valid_step_interval)
            
            loss_avgs.append(float(loss_avg_valid))

    plt.figure(figsize=(10, 6))
    plt.plot(batch_nos, loss_avgs, color='b')
    plt.xlabel('Batch No')
    plt.ylabel('Loss (avg)')
    plt.title(f'{city}')

    plt.savefig(log_file+f"/validation_loss_plot_{city}.png")
    plt.close()