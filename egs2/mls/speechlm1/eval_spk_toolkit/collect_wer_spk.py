import numpy as np
import os
import sys
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--utt_result_file', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=".")
parser.add_argument('--spk_file', type=str, default="")
parser.add_argument('--nbest', type=int, default=1)
args = parser.parse_args()

utt_result_file = args.utt_result_file
output_dir = args.output_dir
spk_file = args.spk_file
nbest = args.nbest
if os.path.exists(spk_file):
    spk_list = open(spk_file, 'r').readlines()
    spk_list = [spk.strip() for spk in spk_list]
    threshold = float(spk_file.split('_')[-1].split('.txt')[0])
else:
    spk_list = []
    threshold = None
    
utt_result = open(utt_result_file, 'r').readlines()

spk_table, utt_wer, threshold_utt_wer = {}, {}, {}
output_utt = open(f'{output_dir}/utt_wer.json', 'w')
for line in utt_result:
    line = line.strip().replace("'", '"')
    data = json.loads(line)
    
    key, wer, weight = data['key'], data['wer'], data['weight']
    spk = key.split('_')[1]
    key = key.split('_sample')[0] # Remove sample idx
    if spk not in spk_table:
        spk_table[spk] = {}
    if key not in spk_table[spk].keys():
        spk_table[spk][key] = {'wer': [], 'weight': []}
        
    if key not in utt_wer:
        utt_wer[key] = {'wer': [], 'weight': []}
    
    if len(spk_list) > 0 and spk in spk_list and key not in threshold_utt_wer:
        threshold_utt_wer[key] = {'wer': [], 'weight': []}
    
    spk_table[spk][key]['wer'].append(wer)
    spk_table[spk][key]['weight'].append(weight)
    utt_wer[key]['wer'].append(wer)
    utt_wer[key]['weight'].append(weight)
    
    if len(spk_list) > 0 and spk in spk_list:
        threshold_utt_wer[key]['wer'].append(wer)
        threshold_utt_wer[key]['weight'].append(weight)
        

        

output = open(f'{output_dir}/spk_wer.json', 'w')
spk_wer = {}
for spk in spk_table.keys():
    avg_wer = []
    for key in spk_table[spk].keys():
        wer_list = np.array(spk_table[spk][key]['wer'])
        weight_list = np.array(spk_table[spk][key]['weight'])
        min_idx = np.argmin(wer_list)
        wer, weight = wer_list[min_idx], weight_list[min_idx]
        avg_wer.append(wer/weight)
    # print(f"Spk {spk}: avg: {avg_wer:.5f}, std: {std_wer:.5f}")
    spk_wer[spk] = {'avg': np.mean(avg_wer), 'std': np.std(avg_wer), 'num': len(spk_table[spk])}
    
json.dump(spk_wer, output, indent=4)

total_wer, total_weight = 0, 0
for key in utt_wer.keys():
    wer_list, weight_list = utt_wer[key]['wer'], utt_wer[key]['weight']
    min_idx = np.argmin(wer_list)
    total_wer += wer_list[min_idx] 
    total_weight += weight_list[min_idx]
    
print(f"Total WER: {total_wer / total_weight}")
print(f"Num spks: {len(spk_table)}")
print(f"Num utts: {len(utt_wer)}")
json.dump(utt_wer, output_utt, indent=4)
if len(spk_list) > 0:
    total_wer, total_weight = 0, 0
    for key in threshold_utt_wer:
        wer_list, weight_list = threshold_utt_wer[key]['wer'], threshold_utt_wer[key]['weight']
        min_idx = np.argmin(wer_list)
        total_wer += wer_list[min_idx] 
        total_weight += weight_list[min_idx]
    print(f"Total WER (threshold={threshold}): {total_wer / total_weight}")
    print(f"Num spks: {len(spk_list)}")
    print(f"Num utts: {len(threshold_utt_wer)}")
output_threshold_utt = open(f'{output_dir}/threshold_utt_wer.json', 'w')
json.dump(threshold_utt_wer, output_threshold_utt, indent=4)

    