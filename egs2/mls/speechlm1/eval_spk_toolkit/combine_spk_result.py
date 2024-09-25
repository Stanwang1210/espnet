import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--wer_json', type=str, default=None)
parser.add_argument('--mos_json', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=".")

args = parser.parse_args()

wer_json = args.wer_json
mos_json = args.mos_json
output_dir = args.output_dir    
output_csv = open(f'{output_dir}/wer_vs_spkmos.csv', 'w')
output_csv.write('SPK,WER,SPKMOS,NUM_SAMPLES\n')
wer_result = json.load(open(wer_json, 'r'))
mos_result = json.load(open(mos_json, 'r'))

spks = wer_result.keys()
for spk in spks:
    wer = wer_result[spk]
    mos = mos_result[spk]
    output_csv.write(f'{spk},{wer["avg"]},{mos["avg"]},{wer["num"]}\n')


