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
lang=$1
ngpu=$2
stage=$3
stop_stage=$4
codec_choice=$5
codec_opts="--codec_choice ESPnet "
expdir=exp_esc


train_set="emilia_${lang}_train"
valid_set="emilia_${lang}_dev"
test_sets="emilia_${lang}_test"
train_set="esc50_train"
valid_set="esc50_dev"
test_sets="esc50_test"
inference_config=conf/decode_asr.yaml

tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used
# codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"
# codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"
# codec_opts="--codec_choice EnCodec "

# ts: true sequence. codec should always in "ts" case
# rm: deduplicated sequence which removes duplicated tokens
src_case="ts"
tgt_case="ts"

fs=16000
if [ ${codec_choice} == "English_EnCodec" ]; then
    codec_opts+=" --codec_hf_model_tag espnet/mls-english_encodec_16k_360epoch"
    asr_config=conf/tuning/train_asr_ebranchformer_english_encodec.yaml
elif [ ${codec_choice} == "Multi_EnCodec" ]; then
    codec_opts+=" --codec_hf_model_tag espnet/mls-multi_encodec_16k_360epoch"
    asr_config=conf/tuning/train_asr_ebranchformer_multi_encodec.yaml
elif [ ${codec_choice} == "Audio_EnCodec" ]; then
    codec_opts+=" --codec_hf_model_tag espnet/mls-audioset_encodec_16k_360epoch"
    asr_config=conf/tuning/train_asr_ebranchformer_audioset_encodec.yaml
elif [ ${codec_choice} == "English_Soundstream" ]; then
    codec_opts+=" --codec_hf_model_tag espnet/mls-english_soundstream_16k"
    asr_config=conf/tuning/train_esc_ebranchformer_english_soundstream.yaml
elif [ ${codec_choice} == "Multi_Soundstream" ]; then
    codec_opts+=" --codec_hf_model_tag espnet/mls-multi_soundstream_16k"
    asr_config=conf/tuning/train_esc_transformer_multi_soundstream.yaml
elif [ ${codec_choice} == "Audio_Soundstream" ]; then
    codec_opts+=" --codec_hf_model_tag espnet/mls-audioset_soundstream_16k"
    asr_config=conf/tuning/train_esc_ebranchformer_audioset_soundstream.yaml
fi
log "${codec_opts}"
./asr2.sh \
    --tokenization_choice "codec" \
    --nj 16 \
    --ngpu ${ngpu} \
    --stage ${stage} \
    --fs ${fs} \
    --stop_stage ${stop_stage} \
    --gpu_inference true \
    --audio_format flac.ark \
    --fs ${fs} \
    --expdir ${expdir} \
    --use_lm false \
    --lang "${lang}" \
    --src_lang ${src_lang} \
    --tgt_lang ${lang} \
    --src_token_type "null" \
    --tgt_token_type "word" \
    --tgt_case ${tgt_case} \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    ${codec_opts} 
