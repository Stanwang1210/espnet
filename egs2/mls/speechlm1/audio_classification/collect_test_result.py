import argparse
import json
from pathlib import Path
parser = argparse.ArgumentParser(description="ESC")
parser.add_argument("--exp_dir", type=str, default="", help="path to the exp directory")
parser.add_argument("--output_file", type=str, default="test_acc.csv", help="path to the output file")
args = parser.parse_args()

fold = 5
tag = args.exp_dir.split("/")[-1]
output_file = Path(args.exp_dir) / f"{tag}_{args.output_file}"
with open(output_file, "w") as out_f:
    for i in range(1, fold + 1): 
        out_f.write(f"Fold_{i}, ")
        
    out_f.write("\n")

    for i in range(1, fold + 1):
        test_json = Path(args.exp_dir) / f"fold_{i}" / "test_output.json"
        test_output = json.load(open(test_json, "r"))
        test_acc = 0
        for k, v in test_output.items():
            if v["label"] == v["pred"]:
                test_acc += 1
        test_acc /= len(test_output)
        out_f.write(f"{test_acc:.3f}, ")
    out_f.write("\n")

