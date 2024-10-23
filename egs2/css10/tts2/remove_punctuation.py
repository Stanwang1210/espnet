
import string
from pathlib import Path
from tqdm import tqdm
import glob
import re


def find_text_files(directory):
    return list(Path(directory).rglob("text"))
def remove_punctuation(input_str):
    return re.sub(r'[^\w\s]', '', input_str).lower().replace('  ', ' ')

# 示例使用

directorys = glob.glob("data")
directorys = [d for d in directorys if '_phn' not in d]



for directory in directorys:
    print(directory)
    text_files = find_text_files(directory)
    for file in tqdm(text_files):
        utt, text = [], []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                utt_id, txt = line.strip().split(maxsplit=1)
                utt.append(utt_id)
                text.append(remove_punctuation(txt))
        with open(file, "w", encoding="utf-8") as f:
            for i in range(len(utt)):
                f.write(f"{utt[i]} {text[i]}\n")