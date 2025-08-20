import random
from collections import Counter, defaultdict
import re
import pickle
import argparse
def is_hex(s):
    return bool(re.match(r'^[0-9a-fA-F]+$', s))

# ['Atlanta', 'WashingtonDC', 'NewYork', 'Seattle', 'LosAngeles', 'Chicago']
parser = argparse.ArgumentParser(description="traj_MOE_MODEL")
parser.add_argument('--city', default='Atlanta', help='City to get distribution')
args = parser.parse_args()
city=args.city
# Step 1: get fisrt location distribution
length_to_counter = defaultdict(Counter)
with open(f"./traj_data/train/{city}_train.txt", 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        trajs = line.split(' ')[3]
        parts = trajs.strip().split(';')
        traj_len = len(parts)
        if traj_len > 0 and parts[0]:
            location, _, _, _, _ = parts[0].split(',')
            first_location = int(location)
            length_to_counter[traj_len][first_location] += 1
# Step 1: construct sampling pool
length_to_sampling_pool = {}
for traj_len, counter in length_to_counter.items():
    pool = []
    for loc, count in counter.items():
        pool.extend([loc] * count)
    length_to_sampling_pool[traj_len] = pool

with open(f"./GCL_data/data_distribution/{city}_length_to_sampling_pool.pkl", "wb") as f:
    pickle.dump(length_to_sampling_pool, f)