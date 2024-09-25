import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=".")
args = parser.parse_args()
csv_file = args.csv
output_dir = args.output_dir
output_name = csv_file.split('/')[-1].split('.')[0]
# 讀取 CSV 文件
df = pd.read_csv(csv_file, header=0)
# print(df.rows)
# 提取 WER 和 MOS 列
wer = df.iloc[1:,1]
mos = df.iloc[1:,2]

# # 計算皮爾森相關係數
correlation, _ = pearsonr(wer, mos)
plt.scatter(wer, mos)
plt.xlabel('WER')
plt.ylabel('UTMOS')
plt.title('WER vs UTMOS')
plt.savefig(f'{output_dir}/{output_name}.png')
print(f"CSV file: {csv_file}")
print(f'Correlation between WER and UTMOS ({output_name}): {correlation}')