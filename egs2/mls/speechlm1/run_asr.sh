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

log "$0 $*"
src_lang=codec
codec_choice=$1
lang=en
codec_opts="--codec_choice ESPnet "
expdir=exp_asr


train_set="mls_${lang}_train_subset"
valid_set="mls_${lang}_dev"
test_sets="mls_${lang}_test"


inference_config=conf/decode_asr.yaml

tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used
# codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"
# codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"
# codec_opts="--codec_choice EnCodec "

# ts: true sequence. codec should always in "ts" case
# rm: deduplicated sequence which removes duplicated tokens
src_case="ts"
tgt_case="ts"

if [ ${codec_choice} == "English_EnCodec" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/mls-english_encodec_16k_360epoch"
    asr_config=conf/tuning/train_asr_ebranchformer_english_encodec.yaml
elif [ ${codec_choice} == "Multi_EnCodec" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/mls-multi_encodec_16k_360epoch"
    asr_config=conf/tuning/train_asr_ebranchformer_multi_encodec.yaml
elif [ ${codec_choice} == "Audio_EnCodec" ]; then
    fs=16000
    codec_opts+=" --codec_hf_model_tag espnet/mls-audioset_encodec_16k_360epoch"
    asr_config=conf/tuning/train_asr_ebranchformer_audioset_encodec.yaml
fi
log "${codec_opts}"
./asr2.sh \
    --tokenization_choice "codec" \
    --nj 8 \
    --ngpu 1 \
    --stage 13 \
    --fs ${fs} \
    --stop_stage 15 \
    --gpu_inference true \
    --audio_format flac.ark \
    --fs ${fs} \
    --expdir ${expdir} \
    --use_lm false \
    --lang "${lang}" \
    --src_lang ${src_lang} \
    --tgt_lang ${lang} \
    --src_token_type "null" \
    --tgt_token_type "char" \
    --tgt_case ${tgt_case} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    ${codec_opts} 
