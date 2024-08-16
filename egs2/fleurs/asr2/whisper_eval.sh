set -e
set -u
set -o pipefail
source /home/stan/miniconda3/envs/espnet_codec/etc/profile.d/conda.sh
conda activate /home/stan/miniconda3/envs/espnet_codec

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}



langs=("ceb_ph" "fil_ph" "gu_in" "jv_id" "kea_cv" "kam_ke" "kn_in" "lo_la" "ln_cd" "luo_ke" "mi_nz" "ny_mw" "sn_zw" "sd_in" "umb_ao" "wo_sn")
dsets=("dev" "test")
codec_choice="ESPnet"
fs=16000
for lang in "${langs[@]}";
do
    for dset in "${dsets[@]}" ;
    do
        log "Evaluating ${codec_choice} ${fs} on ${dset}_${lang}"
        mkdir -p whisper_eval_results/${codec_choice}/${dset}/${lang}

        bash scripts/utils/evaluate_asr.sh \
        --gt_text dump_${fs}/${codec_choice}/${dset}_${lang}/text.ts.${lang} \
        --decode_options "{task: transcribe, beam_size: 1}" \
        "dump_${fs}/${codec_choice}/${dset}_${lang}/wav_resyn.scp" \
        "whisper_eval_results/${codec_choice}/${dset}/${lang}" 
    done
done