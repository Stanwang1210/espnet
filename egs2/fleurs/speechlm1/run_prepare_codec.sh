#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# For environment setup
conda_root=/home/stan/miniconda3
env_name=espnet_speechlm
source ${conda_root}/envs/${env_name}/etc/profile.d/conda.sh
conda activate ${conda_root}/envs/${env_name}


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "$0 $*"
src_lang=codec
tgt_lang=$1
fs=$2
codec_opts="$3"
# expdir=$5
log "${codec_opts}"


train_set=train_"$(echo "${tgt_lang}" | tr - _)"
train_dev=dev_"$(echo "${tgt_lang}" | tr - _)"
test_sets="${train_dev} test_$(echo ${tgt_lang} | tr - _)"

train_config=conf/train_valle.yaml
# train_config=conf/train_multiscale.yaml
inference_config=conf/decode_espnet.yaml

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en
local_data_opts="--trim_all_silence true" # trim all silence in the audio


# ts: true sequence. codec should always in "ts" case
# rm: deduplicated sequence which removes duplicated tokens
src_case="ts"
tgt_case="ts"


./speechlm.sh \
    --task "tts" \
    --data_name fleurs \
    --stage 5 \
    --stop_stage 5 \
    --fs 16000 \
    --ngpu 1 \
    --nj 16 \
    --cleaner "${cleaner}" \
    --nbpe 500 \
    --g2p "${g2p}" \
    --inference_nj 1 \
    --gpu_inference true \
    --local_data_opts "--lang ${tgt_lang}" \
    --audio_format "flac.ark" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    ${codec_opts} 
