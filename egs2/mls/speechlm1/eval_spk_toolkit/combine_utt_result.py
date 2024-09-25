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
output_csv = open(f'{output_dir}/wer_vs_uttspkmos.csv', 'w')
output_csv.write('SPK,WER,UTT_SPKMOS,WEIGHT\n')
wer_result = json.load(open(wer_json, 'r'))
mos_result = json.load(open(mos_json, 'r'))

utts = wer_result.keys()
for utt in utts:
    wer = wer_result[utt]
    mos = mos_result[utt]
    output_csv.write(f'{utt},{wer["wer"]},{mos},{wer["weight"]}\n')


