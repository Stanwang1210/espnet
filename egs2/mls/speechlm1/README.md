# How to use this recipe

## General Pipeline
This recipe provide scripts to run codec robustness experiment.

### Scripts for codec extraction and training

1. Prepare data and extract codec units & reconstructed waveforms from the MLS datasets
```
  bash run_prepare_data.sh
```

2. Run TTS experiment with codec units
```
  run_codec_unit_tts.sh
```

3. Run TTS experiment with codec units
```
  bash run_codec_reconstruct_tts.sh
```

<!-- ### Scripts for reconstructed waveform evaluation 
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
``` -->


## Some changes to the original scripts
1. In `speechlm.sh`, change `dumpdir` to `dumpdir_${fs}` 
