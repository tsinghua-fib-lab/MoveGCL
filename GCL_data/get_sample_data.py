import os
import random
import argparse
parser = argparse.ArgumentParser(description="traj_MOE_MODEL")
parser.add_argument('--city', default='Atlanta', help='City to sample')
args = parser.parse_args()
city_name=args.city
input_dir = "./traj_data/train"
output_dir = "./GCL_data/sampled_data"

os.makedirs(output_dir, exist_ok=True)
filename=city_name+"_train.txt"
input_path = os.path.join(input_dir, filename)
with open(input_path, "r") as f:
    lines = f.readlines()
if len(lines) < 24000:
    print(f"file less than 24000: {filename}")
selected_lines = random.sample(lines, 24000)
with open(os.path.join(output_dir, filename), "w") as f:
    f.writelines(selected_lines)
print(f"处理完成: {filename}")
