import os
from os.path import join
from pathlib import Path
import argparse
import numpy as np
ALL_LANG = ["ceb_ph", "fil_ph", "gu_in", "jv_id", "kea_cv", "kam_ke", "kn_in", "lo_la", "ln_cd", "luo_ke", "mi_nz", "ny_mw", "sn_zw", "sd_in", "umb_ao", "wo_sn"]

EVAL_SETS = ["dev", "test"]
METRICS = ["CER", "WER"]
def main(
    codec_choice: str,
    result_dir: Path,
    output_file: str,
):
    
    assert result_dir.exists(), f"Directory {result_dir} does not exist."
    assert (result_dir / codec_choice).exists(), f"Directory {result_dir / codec_choice} does not exist."
    output_file = Path(f"{codec_choice}_{output_file}")
    avg_result = {}
    with open(output_file, "w") as f:
        f.write("Language (Dev / Test),CER,WER\n")
        for lang in ALL_LANG:
            result_values = {}
            for metric in METRICS:
                for dset in EVAL_SETS:
                    with open(result_dir / codec_choice / dset / lang / f"score_{metric.lower()}" / "result.txt", "r", errors='ignore') as result:
                        result = [l.strip() for l in result.readlines() if "Sum/Avg" in l][0].split()[-3]
                        result_values[f"{dset}_{metric}"] = result
                        if f"{dset}_{metric}" not in avg_result:
                            avg_result[f"{dset}_{metric}"] = []
                        avg_result[f"{dset}_{metric}"].append(float(result))
            result = f"{result_values['dev_CER']} / {result_values['test_CER']},{result_values['dev_WER']} / {result_values['test_WER']}"       
            f.write(f"{lang},{result}\n")
            
        f.write(f"Average, {np.mean(avg_result['dev_CER']):.1f} / {np.mean(avg_result['test_CER']):.1f}, {np.mean(avg_result['dev_WER']):.1f} / {np.mean(avg_result['test_WER']):.1f}\n")
        f.write(f"Std Dev, {np.std(avg_result['dev_CER']):.1f} / {np.std(avg_result['test_CER']):.1f}, {np.std(avg_result['dev_WER']):.1f} / {np.std(avg_result['test_WER']):.1f}\n")
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Convert Kaldi SCP to WAV files.")
    parser.add_argument('--codec_choice', type=str, default="ESPnet")
    parser.add_argument('--result_dir', type=Path, default="whisper_eval_results")
    parser.add_argument('--output_file', type=str, default="whisper_eval_results.csv")
    
    args = parser.parse_args()
    main(**vars(args))
        