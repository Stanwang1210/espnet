set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

current_date=$(date +%Y-%m-%d)
log "Please run the 'run_prepare_data.sh' script before running this script"
# ngpu=2
# stage=3
# stop_stage=3

# codec_choices=(SoundStream EnCodec SoundStream_amuse "EnCodec_amuse")
codec_choices=(English_SoundStream English_EnCodec Multi_SoundStream Multi_EnCodec )
# for codec_choice in "${codec_choices[@]}";
# do 
#     bash run_nar_tts.sh ${codec_choice} ${stage} ${stop_stage} ${ngpu}
# done

stage=7
stop_stage=15
ngpu=1
tag=mls_nar_tts
gpu_cluster=GPU-shared
log_dir=slurm_logs/${current_date}/${tag}
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
            --gpus=v100-32:1 \
            --cpus-per-task=5 \
            --mem=30G \
            --wrap="bash run_nar_tts.sh ${codec_choice} ${stage} ${stop_stage} ${ngpu}"
done