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
asr_config=$3
codec_opts="$4"
expdir=$5
log "${codec_opts}"


train_set=train_"$(echo "${tgt_lang}" | tr - _)"
train_dev=dev_"$(echo "${tgt_lang}" | tr - _)"
test_sets="${train_dev} test_$(echo ${tgt_lang} | tr - _)"


inference_config=conf/decode_ctc0.3.yaml

tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used
# codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"
# codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"
# codec_opts="--codec_choice EnCodec "

# ts: true sequence. codec should always in "ts" case
# rm: deduplicated sequence which removes duplicated tokens
src_case="ts"
tgt_case="ts"


./asr2.sh \
    --tokenization_choice "codec" \
    --nj 8 \
    --ngpu 1 \
    --stage 6 \
    --stop_stage 15 \
    --gpu_inference true \
    --audio_format flac.ark \
    --fs ${fs} \
    --expdir ${expdir} \
    --use_lm false \
    --lang "${tgt_lang}" \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "null" \
    --tgt_token_type "char" \
    --tgt_case ${tgt_case} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --tgt_bpe_train_text "data/${train_set}/text" \
    ${codec_opts} 
