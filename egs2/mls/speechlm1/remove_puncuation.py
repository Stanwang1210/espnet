import string
from pathlib import Path
from tqdm import tqdm
import glob
def find_text_files(directory):
    return list(Path(directory).rglob("text"))
def remove_punctuation(input_str):
	# 創建一個翻譯表，將所有標點符號映射到 None
	translator = str.maketrans('', '', string.punctuation)
	# 使用翻譯表移除標點符號
	return input_str.translate(translator).lower()

# 示例使用

directorys = glob.glob("/ocean/projects/cis210027p/swang26/espnet_codec/egs2/mls/speechlm1/dump_16000/audio_raw_tts_emilia")

for directory in directorys:
    print(directory)
# directory = "/ocean/projects/cis210027p/swang26/espnet_codec/esgs2/mls/speechlm1/data/emilia_fr_test"
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
# cleaned_str = remove_punctuation(example_str)
# print("Original:", example_str)
# print("Cleaned:", cleaned_str)