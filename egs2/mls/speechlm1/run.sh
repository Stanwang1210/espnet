#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
conda_root=/ocean/projects/cis210027p/swang26/miniconda3
env_name=espnet_codec
source ${conda_root}/envs/${env_name}/etc/profile.d/conda.sh
conda activate ${conda_root}/envs/${env_name}
lang=$1
stage=$2
stop_stage=$3 
codec_choice=${4:-"SoundStream"}
ngpu=${5:-4}
data_split="full" # one of full 1h 10h
local_data_opts="--lang ${lang} --data_split ${data_split} --stage 1 "

train_set="mls_${lang}_train_subset"
valid_set="mls_${lang}_dev"
test_sets="mls_${lang}_dev mls_${lang}_test"

expdir=exp_ar_tts
bpe_opts="--nbpe 200"
train_config=conf/train_valle.yaml
task="tts"
data_name="mls"
data_combo_name="${task}_${data_name}_${lang}"
codec_opts="--codec_choice ESPnet "
inference_nj=4

if [ ${codec_choice} == "SoundStream" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/libritts_soundstream16k"
    inference_config=conf/decode_espnet_libri_soundstream.yaml
elif [ ${codec_choice} == "EnCodec" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/libritts_encodec_16k"
    inference_config=conf/decode_espnet_libri_encodec.yaml
elif [ ${codec_choice} == "SoundStream_amuse" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/amuse_soundstream16k"
    inference_config=conf/decode_espnet_amuse_soundstream.yaml
elif [ ${codec_choice} == "EnCodec_amuse" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/amuse_encodec_16k"
    inference_config=conf/decode_espnet_amuse_encodec.yaml
elif [ ${codec_choice} == "EnCodec_original" ]; then
    fs=24000
    inference_config=conf/decode_encodec.yaml

else
    echo "Unknown codec choice"
    exit 1
fi
# NOTE(Jinchuan): This script is only to prepare data. End at stage 5
./speechlm.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --local_data_opts "${local_data_opts}" \
    --task ${task} \
    --expdir ${expdir} \
    --data_name ${data_name} \
    --data_combo_name "${data_combo_name}" \
    --fs ${fs} \
    --ngpu ${ngpu} \
    --nj 64 \
    --inference_nj ${inference_nj} \
    --gpu_inference true \
    --cleaner None \
    --train_config ${train_config} \
    --audio_format "flac.ark" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    ${bpe_opts} ${codec_opts} 