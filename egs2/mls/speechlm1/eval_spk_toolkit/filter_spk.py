import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mos_json', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=".")
parser.add_argument('--threshold', type=float, default=3.0)

args = parser.parse_args()

mos_json = args.mos_json
output_dir = args.output_dir   
threshold = args.threshold
mos_result = json.load(open(mos_json, 'r'))
output_file = open(f'{output_dir}/spk_mos_gt_{threshold}.txt', 'w')
for spk in mos_result.keys():
    mos = mos_result[spk]['avg']
    if float(mos) > threshold:
        # print(f"Spk: {spk}, MOS: {mos}")    
        output_file.write(f"{spk}\n")