import json
import os
from pathlib import Path
import sys
import string
import subprocess
# for line in lines:
#     input_file = line.strip()
#     output_file = input_file.replace(".mp3", ".wav")
    
#     # 使用 ffmpeg 進行轉換和重新採樣
#     subprocess.run(["ffmpeg", "-i", input_file, "-ar", "16000", output_file])
def remove_punctuation(input_str):
	# 創建一個翻譯表，將所有標點符號映射到 None
	translator = str.maketrans('', '', string.punctuation)
	# 使用翻譯表移除標點符號
	return input_str.translate(translator).lower()
lang = sys.argv[1]
lang_table = {
    'de': 'german',
    'fr': 'french'
}
language = lang_table[lang]
clips_table = {
    'de': f'commonvoice_{language}/cv-corpus-15.0-delta-2023-09-08/de/clips',
    'fr': f'commonvoice_{language}/cv-corpus-16.1-delta-2023-12-06/fr/clips'
}
json_path = f"commonvoice_{language}/{language}_test.json"
output_path = f"data/commonvoice_{lang}_test"

os.makedirs(output_path, exist_ok=True)


data = json.load(open(json_path, 'r', encoding='utf-8'))
utt2spk = open(f"{output_path}/utt2spk", 'w')
text_file = open(f"{output_path}/text", 'w')
wavscp = open(f"{output_path}/wav.scp", 'w')

wav_root = clips_table[lang]
for d in data:
    
    text = remove_punctuation(d['text'])
    speaker = d['speaker'][:10]
    wav_path = d['wav_path']
    uttid = f"{speaker}-{wav_path.replace('.mp3', '')}"
    wav_path = f"{wav_root}/{d['wav_path']}"
    
    assert os.path.exists(wav_path), f"{wav_path} not found"
    
    output_file = wav_path.replace(".mp3", ".wav")
    
    # 使用 ffmpeg 進行轉換和重新採樣
    # subprocess.run(["ffmpeg", "-i", wav_path, "-ar", "16000", output_file])
    utt2spk.write(f"{uttid} {speaker}\n")
    wavscp.write(f"{uttid} sox {output_file} -r 16000 -t wav -c 1 -b 16 - |\n")
    text_file.write(f"{uttid} {text}\n")

    
