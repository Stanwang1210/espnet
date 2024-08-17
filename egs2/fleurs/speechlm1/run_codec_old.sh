#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
source /home/stan/miniconda3/envs/espnet_codec/etc/profile.d/conda.sh
conda activate /home/stan/miniconda3/envs/espnet_codec

langs=("ceb_ph" "fil_ph" "gu_in" "jv_id" "kea_cv" "kam_ke" "kn_in" "lo_la" "ln_cd" "luo_ke" "mi_nz" "ny_mw" "sn_zw" "sd_in" "umb_ao" "wo_sn")
# langs=("fil_ph" "gu_in" "jv_id" "kea_cv" "kam_ke" "kn_in" "lo_la" "ln_cd" "luo_ke" "mi_nz" "ny_mw" "sn_zw" "sd_in" "umb_ao" "wo_sn")

for lang in "${langs[@]}"
do
    src_lang=codec
    tgt_lang=${lang}


    train_set=train_"$(echo "${tgt_lang}" | tr - _)"
    train_dev=dev_"$(echo "${tgt_lang}" | tr - _)"
    test_sets="${train_dev} test_$(echo ${tgt_lang} | tr - _)"


    asr_config=conf/tuning/train_asr_ebranchformer_ESPnet.yaml
    inference_config=conf/decode_ctc0.3.yaml

    tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used
    # codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"
    codec_opts="--codec_choice EnCodec"

    # ts: true sequence. codec should always in "ts" case
    # rm: deduplicated sequence which removes duplicated tokens
    src_case="ts"
    tgt_case="ts"


    ./asr2.sh \
        --tokenization_choice "codec" \
        --nj 12 \
        --ngpu 1 \
        --stage 3 \
        --stop_stage 5 \
        --audio_format flac.ark \
        --fs 24000 \
        --use_lm false \
        --lang "${tgt_lang}" \
        --src_lang ${src_lang} \
        --tgt_lang ${tgt_lang} \
        --src_token_type "null" \
        --tgt_token_type "bpe" \
        --tgt_case ${tgt_case} \
        --asr_config "${asr_config}" \
        --inference_config "${inference_config}" \
        --train_set "${train_set}" \
        --valid_set "${train_dev}" \
        --test_sets "${test_sets}" \
        --tgt_bpe_train_text "data/${train_set}/text" \
        --local_data_opts "--lang ${tgt_lang}" \
        ${codec_opts} "$@"
done
