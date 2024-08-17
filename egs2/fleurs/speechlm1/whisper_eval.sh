set -e
set -u
set -o pipefail
conda_root=$(conda info --base)
env_name=espnet_codec
source ${conda_root}/envs/${env_name}/etc/profile.d/conda.sh
conda activate ${conda_root}/envs/${env_name}

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}



langs=("ceb_ph" "fil_ph" "gu_in" "jv_id" "kea_cv" "kam_ke" "kn_in" "lo_la" "ln_cd" "luo_ke" "mi_nz" "ny_mw" "sn_zw" "sd_in" "umb_ao" "wo_sn")
dsets=("dev" "test")
if [ ${codec_choice} == "ESPnet" ]; then
    fs=16000
elif [ ${codec_choice} == "EnCodec" ]; then
    fs=24000
else
    echo "Unknown codec choice"
    exit 1
fi
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