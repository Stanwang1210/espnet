#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
# export PHONEMIZER_ESPEAK_LIBRARY=/ocean/projects/cis210027p/swang26/espnet_codec/egs2/mls/speechlm1/espeak-ng/lib/libespeak-ng.so



# eval_set="${lang}_eval1_phn commonvoice_${lang}_test_phn"
stage=6
stop_stage=6
codec_choice=Audio_Soundstream
codec_opts="--codec_choice ESPnet "
if [ ${codec_choice} == "English_Soundstream" ]; then
    codec_opts+=" --codec_hf_model_tag espnet/mls-english_soundstream_16k"
    vocoder_file=espnet/mls-english_soundstream_16k
    inference_config=conf/decode_espnet_english_soundstream.yaml
elif [ ${codec_choice} == "Multi_Soundstream" ]; then
    codec_opts+=" --codec_hf_model_tag espnet/mls-multi_soundstream_16k"
    vocoder_file=espnet/mls-multi_soundstream_16k
    inference_config=conf/decode_espnet_multi_soundstream.yaml
elif [ ${codec_choice} == "Audio_Soundstream" ]; then
    codec_opts+=" --codec_hf_model_tag espnet/mls-audioset_soundstream_16k"
    vocoder_file=espnet/mls-audioset_soundstream_16k
    inference_config=conf/decode_espnet_audioset_soundstream.yaml
fi
# put your vocoder file and vocoder config file here
# vocoder can be trained from
# https://github.com/kan-bayashi/ParallelWaveGAN
# vocoder_file=vocoder/vocoder.pkl

# duration information
teacher_dumpdir=teacher_dumpdir
cleaner=None
g2p=None
langs=(en es de)
# langs=(en)
for lang in ${langs[@]}; 
do
    train_set="${lang}_dev"
    train_dev="${lang}_dev"
    eval_set="${lang}"
    # train_set="${lang}_tr_no_dev_phn"
    # train_dev="${lang}_dev_phn"
    # eval_set="${lang}_eval1_phn"
    ./tts2.sh \
        --nj 1 \
        --ngpu 1 \
        --dumpdir dump_audio \
        --inference_nj 4 \
        --gpu_inference true \
        --inference_model valid.loss.best.pth \
        --fs 16000 --n_shift 320 --n_fft 1280 \
        --lang ${lang} \
        --src_token_type word \
        --audio_format wav.ark \
        --g2p ${g2p} \
        --cleaner ${cleaner} \
        --stage ${stage} \
        --stop_stage ${stop_stage} \
        --train_config conf/train_fastspeech2.yaml \
        --train_set "${train_set}" \
        --valid_set "${train_dev}" \
        --test_sets "${eval_set}" \
        --write_collected_feats true \
        --vocoder_file ${vocoder_file} \
        --use_spk_embed false \
        --skip_train true \
        ${codec_opts} \
        --srctexts "data/${train_set}/text" 
done