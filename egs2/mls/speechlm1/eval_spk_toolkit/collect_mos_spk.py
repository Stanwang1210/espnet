import numpy as np
import os
import sys
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--utt_result_file', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=".")
args = parser.parse_args()

utt_result_file = args.utt_result_file
output_dir = args.output_dir
utt_result = open(utt_result_file, 'r').readlines()

spk_table, utt_mos = {}, {}
output_utt = open(f'{output_dir}/utt_mos.json', 'w')
for line in utt_result:
    line = line.strip().replace("'", '"')
    data = json.loads(line)
    
    key, mos = data['key'], data['utmos']
    utt_mos[key] = mos
    spk = key.split('_')[1]
    
    if spk not in spk_table:
        spk_table[spk] = []
    spk_table[spk].append(mos)
json.dump(utt_mos, output_utt, indent=4)
    
output_spk = open(f'{output_dir}/spk_mos.json', 'w')

spk_mos = {}
for spk in spk_table:
    mos = np.array(spk_table[spk])
    avg_mos = np.mean(mos)
    std_mos = np.std(mos)   
    # print(f"Spk {spk}: avg: {avg_mos:.5f}, std: {std_mos:.5f}")
    spk_mos[spk] = {'avg': avg_mos, 'std': std_mos, 'num': len(mos)}
    
json.dump(spk_mos, output_spk, indent=4)
    