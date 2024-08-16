set -e
set -u
set -o pipefail
source /home/stan/miniconda3/envs/espnet_codec/etc/profile.d/conda.sh
conda activate /home/stan/miniconda3/envs/espnet_codec

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

result_dir=versa_results
mkdir -p ${result_dir}
langs=("ceb_ph" "fil_ph" "gu_in" "jv_id" "kea_cv" "kam_ke" "kn_in" "lo_la" "ln_cd" "luo_ke" "mi_nz" "ny_mw" "sn_zw" "sd_in" "umb_ao" "wo_sn")
# langs=("fil_ph")
dsets=("dev" "test")
fs=16000
codec_choice=ESPnet
for set in "${dsets[@]}" ;
do
    for lang in "${langs[@]}"
    do
        log "Evaluating ${codec_choice} ${fs} on ${set}_${lang}"
        codec_dir=dump_${fs}/${codec_choice}/${set}_${lang}
        audio_dir=dump_${fs}/audio_raw/${set}_${lang}
        log "Codec dir: ${codec_dir}"
        log "Audio dir: ${audio_dir}"
        
        if [ ! -d ${codec_dir}/wav_resyn_${codec_choice} ]; then
            log "Resynthesizing ${set}_${lang} with ${codec_choice} ${fs}"
            python3 convert_kaldi2wav.py \
            --scp_file ${codec_dir}/wav_resyn_${codec_choice}.scp \
            --output_dir ${codec_dir}/wav_resyn_${codec_choice}
        fi
        
        if [ ! -d ${audio_dir}/wav_ground_${codec_choice} ]; then
            log "Ground Truth ${set}_${lang} with ${codec_choice} ${fs}"
            python3 convert_kaldi2wav.py \
            --scp_file ${audio_dir}/wav.scp \
            --output_dir ${audio_dir}/wav_ground_${codec_choice}
        fi



        log "Evaluate wav files at ${codec_dir}/wav_resyn_${codec_choice}"

        python3 versa/versa/bin/scorer.py \
        ${codec_dir}/wav_resyn_${codec_choice} \
        --score_config versa/egs/codec_16k_cpu.yaml \
        --gt ${audio_dir}/wav_ground_${codec_choice} \
        --output results/${codec_choice}_${set}_${lang}.txt

        log "Done evaluating ${codec_choice} ${fs} on ${set}_${lang}"
        log "Results are saved at ${result_dir}/${codec_choice}_${set}_${lang}.txt"

    done
done