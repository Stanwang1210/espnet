#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
# For environment setup
# Can comment out the following two lines if you have already activated conda
conda_root=$(conda info --base)
env_name=espnet_codec
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
train_config=$3
codec_opts="$4"
expdir=$5
local_data_opts="--lang ${tgt_lang} "
log "${codec_opts}"


train_set=train_"$(echo "${tgt_lang}" | tr - _)"
valid_set=dev_"$(echo "${tgt_lang}" | tr - _)"
test_sets="${valid_set} test_$(echo ${tgt_lang} | tr - _)"


inference_config=conf/decode_ctc0.3.yaml

tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used
# codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"
# codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"
# codec_opts="--codec_choice EnCodec "

# ts: true sequence. codec should always in "ts" case
# rm: deduplicated sequence which removes duplicated tokens
src_case="ts"
tgt_case="ts"


./speechlm.sh \
    --task "tts" \
    --data_name fleurs \
    --expdir ${expdir} \
    --lang "${tgt_lang}" \
    --fs ${fs} \
    --ngpu 1 \
    --nj 16 \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --inference_nj 1 \
    --gpu_inference true \
    --local_data_opts "${local_data_opts}" \
    --audio_format "flac.ark" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    ${codec_opts}