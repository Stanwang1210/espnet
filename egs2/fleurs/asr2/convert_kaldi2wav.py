import kaldiio
import soundfile as sf
import numpy as np
import os

import argparse


def process_scp_file(scp_path, output_dir):
    # Load the SCP file using kaldiio
    wav_scp_file = f"{output_dir}/../wav_resyn.scp"
    wav_scp= open(wav_scp_file, "w")
    with kaldiio.ReadHelper(f'scp:{scp_path}') as reader:
        for key, (sample_rate, wav) in reader:
            output_path = f"{output_dir}/{key}.wav"
            sf.write(output_path, wav, sample_rate)
            wav_scp.write(f"{key} {output_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert Kaldi SCP to WAV files.")
    parser.add_argument('--scp_file', type=str, required=True, help="Path to the SCP file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the WAV files.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    process_scp_file(args.scp_file, args.output_dir)

if __name__ == "__main__":
    main()