import random
from collections import Counter, defaultdict
import re
import pickle
import os
import argparse
parser = argparse.ArgumentParser(description="traj_MOE_MODEL")
parser.add_argument('--city', default='Atlanta', help='City to sample')
args = parser.parse_args()
city_name=args.city

def is_hex(s):
    return bool(re.match(r'^[0-9a-fA-F]+$', s))

city_to_gen = ['Atlanta', 'WashingtonDC', 'NewYork', 'Seattle', 'LosAngeles', 'Chicago']
for city_now in [city_name]:
    for city in city_to_gen:
        with open(f"./GCL_data/data_distribution/{city}_length_to_sampling_pool.pkl", "rb") as f:
            length_to_sampling_pool = pickle.load(f)

        existing_lengths = sorted(length_to_sampling_pool.keys())

        os.makedirs(f"./GCL_data/replaced_first_loc_data/{city_now}", exist_ok=True)

        input_data = f"./GCL_data/sampled_data/{city_now}_train.txt"
        output_data = f"./GCL_data/replaced_first_loc_data/{city_now}/{city}_gen.txt"
        user_id_map = {}
        user_id_counter = 0
        with open(input_data, 'r', encoding='utf-8') as fin, open(output_data, 'w', encoding='utf-8') as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                userid, rg, entropy, trajs = parts[0], parts[1], parts[2], parts[3]
                traj_points = trajs.strip().split(';')
                traj_len = len(traj_points)

                if is_hex(userid):
                    userid_decimal = str(int(userid, 16))
                else:
                    userid_decimal = userid

                if userid_decimal not in user_id_map:
                    user_id_map[userid_decimal] = user_id_counter
                    user_id_counter += 1
                new_user_id = user_id_map[userid_decimal]

                if traj_len > 0 and traj_points[0]:
                    if traj_len in length_to_sampling_pool:
                        sampling_pool = length_to_sampling_pool[traj_len]
                    else:
                        nearest_len = min(existing_lengths, key=lambda x: abs(x - traj_len))
                        sampling_pool = length_to_sampling_pool[nearest_len]

                    first_point_parts = traj_points[0].split(',')
                    first_point_parts[0] = str(random.choice(sampling_pool))
                    traj_points[0] = ','.join(first_point_parts)

                    for i in range(1, traj_len):
                        point_parts = traj_points[i].split(',')
                        if len(point_parts) >= 1:
                            point_parts[0] = '0'
                            traj_points[i] = ','.join(point_parts)

                new_traj = ';'.join(traj_points)
                new_line = f"{new_user_id} {rg} {entropy} {new_traj}\n"
                fout.write(new_line)
