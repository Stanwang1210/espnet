#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
# conda_root=/ocean/projects/cis210027p/sws
export PHONEMIZER_ESPEAK_LIBRARY=/ocean/projects/cis210027p/swang26/espnet_codec/egs2/mls/speechlm1/espeak-ng/lib/libespeak-ng.so
langs=(en es de fr nl)
dsets=(dev test)
langs=(en )
dsets=(test)
lang=$1
ngpu=$2
stage=$3
stop_stage=$4
codec_choice=$5
local_data_opts="--lang ${lang} --stage 1 "

train_set="emilia_${lang}_train"
valid_set="emilia_${lang}_dev"
test_sets="mls_${lang}_test emilia_${lang}_test "
# train_set="commonvoice_train_${lang}"
# valid_set="commonvoice_dev_${lang}"
# test_sets="emilia_${lang}_test commonvoice_${lang}_test"
# test_sets="test_clean"
# for l in ${langs[@]}; do
#     for dset in ${dsets[@]}; do
#         test_sets+="emilia_${l}_${dset} "
#     done
# done

expdir=exp_ar_tts_phn_emilia
bpe_opts="--nbpe 200"
train_config=conf/train_valle.yaml
task="tts"
data_name="emilia"
data_combo_name="${task}_${data_name}_${lang}"
codec_opts="--codec_choice ESPnet "
inference_nj=4

if [ ${codec_choice} == "English_EnCodec" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/mls-english_encodec_16k_360epoch"
    inference_config=conf/decode_espnet_english_encodec.yaml
elif [ ${codec_choice} == "Multi_EnCodec" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/mls-multi_encodec_16k_360epoch"
    inference_config=conf/decode_espnet_multi_encodec.yaml
elif [ ${codec_choice} == "Audio_EnCodec" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/mls-audioset_encodec_16k_360epoch"
    inference_config=conf/decode_espnet_audio_encodec.yaml
elif [ ${codec_choice} == "English_Soundstream" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/mls-english_soundstream_16k"
    inference_config=conf/decode_espnet_english_soundstream.yaml
elif [ ${codec_choice} == "Multi_Soundstream" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/mls-multi_soundstream_16k"
    inference_config=conf/decode_espnet_multi_soundstream.yaml
elif [ ${codec_choice} == "Audio_Soundstream" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/mls-audioset_soundstream_16k"
    inference_config=conf/decode_espnet_audio_soundstream.yaml
elif [ ${codec_choice} == "EnCodec_original" ]; then
    fs=24000
    codec_opts="--codec_choice EnCodec "
    inference_config=conf/decode_encodec.yaml

else
    echo "Unknown codec choice"
    exit 1
fi

if [ ${lang} == "en" ]; then
    g2p="g2p_en "
elif [ ${lang} == "es" ]; then
    g2p="espeak_ng_spanish "
elif [ ${lang} == "de" ]; then
    g2p="espeak_ng_german "
elif [ ${lang} == "fr" ]; then
    g2p="espeak_ng_french "
elif [ ${lang} == "nl" ]; then
    g2p="espeak_ng_dutch "
else
    echo "Unknown language"
    exit 1
fi
# NOTE(Jinchuan): This script is only to prepare data. End at stage 5
./speechlm.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --local_data_opts "${local_data_opts}" \
    --inference_model valid.total_count.best.pth \
    --inference_config ${inference_config} \
    --task ${task} \
    --expdir ${expdir} \
    --data_name ${data_name} \
    --data_combo_name "${data_combo_name}" \
    --fs ${fs} \
    --ngpu ${ngpu} \
    --lang ${lang} \
    --nj 16 \
    --inference_nj ${inference_nj} \
    --gpu_inference true \
    --cleaner None \
    --lang ${lang} \
    --g2p ${g2p} \
    --train_config ${train_config} \
    --audio_format "flac.ark" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    --nbest 10 \
    ${bpe_opts} ${codec_opts} 
    # --skip_train true \