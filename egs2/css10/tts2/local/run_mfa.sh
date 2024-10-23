#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=de # fr, en, de
fs=16000
n_shift=320
if [ ${lang} == "en" ]; then
    language="english"                     # The language type of corpus.
    acoustic_model="english_mfa"    # MFA Acoustic model.
    dictionary="english_us_mfa"     # MFA Dictionary.
    g2p_model="g2p_en"
    cleaner=tacotron
elif [ ${lang} == "es" ]; then
    g2p="espeak_ng_spanish "
elif [ ${lang} == "de" ]; then
    language="german"                     # The language type of corpus.
    acoustic_model="german_mfa"    # MFA Acoustic model.
    dictionary="german_mfa"     # MFA Dictionary.
    g2p_model="espeak_ng_german"
    cleaner=transliteration_cleaners
elif [ ${lang} == "fr" ]; then
    language="french"                     # The language type of corpus.
    acoustic_model="french_mfa"    # MFA Acoustic model.
    dictionary="french_mfa"     # MFA Dictionary.
    g2p_model="espeak_ng_french"
    cleaner=transliteration_cleaners
elif [ ${lang} == "nl" ]; then
    g2p="espeak_ng_dutch "
else
    echo "Unknown language"
    exit 1
fi

split_sets="${lang}_tr_no_dev ${lang}_dev ${lang}_eval1 "
# split_sets="commonvoice_fr_test"
./scripts/utils/mfa.sh \
    --language ${language}  \
    --train true \
    --acoustic_model ${acoustic_model} \
    --g2p_model ${g2p_model} \
    --dictionary ${dictionary} \
    --samplerate ${fs} \
    --hop-size ${n_shift} \
    --cleaner ${cleaner} \
    --stage 1 \
    --stop_stage 5 \
    --split_sets "${split_sets}" \
    --clean_temp false \
    "$@"
