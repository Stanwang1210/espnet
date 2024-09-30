set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
current_date=$(date +%Y-%m-%d)
log "Please run the 'run_prepare_data.sh' script before running this script"
stage=6
stop_stage=15
ngpu=4
tag=mls_ar_tts
lang=all
gpu_cluster=GPU-shared
log_dir=slurm_logs/${current_date}/${tag}
codec_choices=(English_SoundStream English_EnCodec Multi_SoundStream Multi_EnCodec )
mkdir -p ${log_dir}
for codec_choice in "${codec_choices[@]}";
do 
     sbatch  \
            --job-name=${tag}_${codec_choice} \
            --output=${log_dir}/${tag}_${codec_choice}.txt \
            --error=${log_dir}/${tag}_${codec_choice}.err \
            --time=2-00:00:00 \
            --partition=${gpu_cluster} \
            --qos=high_priority \
            --gpus=v100-32:4 \
            --cpus-per-task=10 \
            --mem=60G \
            --wrap="bash run.sh ${lang} ${stage} ${stop_stage} ${codec_choice} ${ngpu}"
    log "Finish extract codec ${codec_choice}"
done