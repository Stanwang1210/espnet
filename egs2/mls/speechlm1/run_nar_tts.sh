#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
# For environment setup
# Can comment out the following two lines if you have already activated conda
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}



codec_choice=$1
stage=$2
stop_stage=$3
ngpu=${4:-4}
n_fft=1024
n_shift=256
lang=all
train_set="mls_${lang}_train"
valid_set="mls_${lang}_dev"
test_sets="mls_${lang}_dev mls_${lang}_test"

train_config=conf/tuning/train_vits.yaml
inference_config=conf/tuning/decode_vits.yaml
expdir=exp_nar_tts
task="tts"
data_name="mls"
data_combo_name="${task}_${data_name}"
cleaner=None
g2p=g2p_en_no_space # or g2p_en
local_data_opts="--trim_all_silence true" # trim all silence in the audio

codec_opts="--codec_choice ESPnet"
if [ ${codec_choice} == "SoundStream" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/libritts_soundstream16k"
elif [ ${codec_choice} == "EnCodec" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/libritts_encodec_16k"
elif [ ${codec_choice} == "SoundStream_amuse" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/amuse_soundstream16k"
elif [ ${codec_choice} == "EnCodec_amuse" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/amuse_encodec_16k"
else
    echo "Unknown codec choice"
    exit 1
fi

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav.ark "
else
    opts="--audio_format wav.ark "
fi

./tts.sh \
    --ngpu ${ngpu} \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --nj 16 \
    --lang ${lang} \
    --feats_type raw \
    --local_data_opts "${local_data_opts}" \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length null \
    --use_spk_embed true \
    --token_type char \
    --cleaner "${cleaner}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --spk_embed_gpu_inference true \
    --expdir ${expdir} \
    --data_combo_name "${data_combo_name}" \
    --tts_task gan_tts \
    --feats_extract linear_spectrogram \
    --feats_normalize none \
    --inference_model latest.pth \
    ${codec_opts} \
    ${opts} 
# done