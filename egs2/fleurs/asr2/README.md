# How to use this recipe

## General Pipeline
This recipe provide scripts to run codec robustness experiment.

### Scripts for codec extraction and training
1. Extract codec units and codec reconstructed waveforms from the FLEURS datasets
```
  bash run_all_prepare_codec.sh
```

2. Run ASR experiment with codec units
```
  bash run_all_codec.sh
```

3. Run ASR experiment with codec reconstructed waveforms
```
  bash run_all_codec_reconstruch.sh
```

### Scripts for reconstructed waveform evaluation 
1. Signal level & Speaker similarity evaluation 
```
  git clone https://github.com/shinjiwlab/versa.git
  cd versa
  pip install .
  bash versa_evaluate.sh
```

2. Content level evaluation
```
  bash whisper_eval.sh
```

### Note for all scripts:
Need to specify the `codec_choice` in 
- `run_all_prepare_codec.sh`
- `run_all_codec.sh`
- `run_all_codec_reconstruch.sh`
- `versa_evaluate.sh` 
- `whisper_eval.sh`

If using the conda, specify the `env_name` in 
- `run_prepare_codec.sh` 
- `run_codec.sh`
- `run_codec_reconstruct.sh`


## Some changes to the original scripts
1. In `asr2.sh` and `asr_reconstruct.sh`, change `dumpdir` to `dumpdir_${fs}` 
2. In `asr2.sh`, set `--dump_audio` to `true`
3. In `asr_reconstruct.sh`, change `ref_text_files_str` from `"text "` to `"text.${tgt_case}.${lang}` "
4. In `asr_reconstruct.sh` stage 10-12, change `_scp` from `wav.scp` to `wav_resyn_$${codec_choice}.scp`
5. In `asr2.sh` and `asr_reconstruct.sh`, change `lm_train.txt` and `token_train.txt` to `${lang}_lm_train.txt` and `${lang}_token_train.txt`.