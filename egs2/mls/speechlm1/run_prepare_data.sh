set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
conda_root=/ocean/projects/cis210027p/swang26/miniconda3
env_name=espnet_codec_robustness
source ${conda_root}/envs/${env_name}/etc/profile.d/conda.sh
conda activate ${conda_root}/envs/${env_name}
current_date=$(date +%Y-%m-%d)
tag=mls_prepare
log "Please set the ngpu here"
ngpu=4
langs=(es en fr nl de)


cpu_cluster=RM-shared
gpu_cluster=GPU-small
process_job_ids=()
# for lang in "${langs[@]}"
# do
#     stage=1
#     stop_stage=1
#     log_dir=slurm_logs/${current_date}/${tag}
#     mkdir -p ${log_dir}
#     log "Start processing data ${lang}"
#     job_id=$(sbatch  --job-name=prepare_data_${tag}_${lang} \
#             --output=${log_dir}/prepare_data_${lang}.txt \
#             --error=${log_dir}/prepare_data_${lang}.err \
#             --time=2-00:00:00 \
#             --partition=${cpu_cluster} \
#             --qos=high_priority \
#             --cpus-per-task=12 \
#             --mem=23G \
#             --wrap="bash run.sh ${lang} ${stage} ${stop_stage}")
#     process_job_ids+=(${job_id##* })

# done

subsample_data_job_ids=()
langs=(en)
datadir="data"
# --dependency=afterok:${process_job_ids[*]}
# for lang in "${langs[@]}"
# do 
#     train_set="mls_$(echo ${lang} | tr - _)_train"
#     tag=mls_subsample
#     log_dir=slurm_logs/${current_date}/${tag}
#     mkdir -p ${log_dir}
#     log "Start subsampling data ${lang}"
#     job_id=$(sbatch   \
#             --job-name=${tag}_${lang} \
#             --output=${log_dir}/${tag}_${lang}.txt \
#             --error=${log_dir}/${tag}_${lang}.err \
#             --time=00:10:00 \
#             --partition=${cpu_cluster} \
#             --qos=high_priority \
#             --cpus-per-task=12 \
#             --mem=23G \
#             --wrap="bash subsample_data.sh ${lang} ${datadir} ${train_set}")
#     subsample_data_job_ids+=(${job_id##* })
# done

# log "Start combining data"
# combine_data_job_ids=$(
#     sbatch  --dependency=afterok:${subsample_data_job_ids[*]} \
#             --job-name=${tag}_${lang} \
#             --output=${log_dir}/${tag}_${lang}.txt \
#             --error=${log_dir}/${tag}_${lang}.err \
#             --time=2-00:00:00 \
#             --partition=${cpu_cluster} \
#             --qos=high_priority \
#             --cpus-per-task=12 \
#             --mem=23G \
#             --wrap="bash combine_data.sh ${langs[@]}"
#     )
# log "Finish combining data"

stage=2
stop_stage=2
lang="en"
langs=(es fr nl de)
process_audio_job_ids=()
tag=mls_audio_prepare
log_dir=slurm_logs/${current_date}/${tag}
mkdir -p ${log_dir}
log "Start processing audio data "
process_audio_job_ids=()
# codec_choices=(English_SoundStream )
# for codec_choice in "${codec_choices[@]}";
# do
#     for lang in "${langs[@]}"
#     do
#         job_id=$(sbatch  \
#                 --job-name=${tag}_${lang} \
#                 --output=${log_dir}/${tag}_${lang}.txt \
#                 --error=${log_dir}/${tag}_${lang}.err \
#                 --time=00:45:00 \
#                 --partition=${cpu_cluster} \
#                 --qos=high_priority \
#                 --cpus-per-task=64 \
#                 --mem=120G \
#                 --wrap="bash run.sh ${lang} ${stage} ${stop_stage} ${codec_choice} 1 "
#     )
#     done
#     process_audio_job_ids+=(${job_id##* })
# done
# log "Finish processing audio data "

stage=5
stop_stage=5
lang="nl"
ngpu=4
tag=mls_codec_prepare
log_dir=slurm_logs/${current_date}/${tag}
mkdir -p ${log_dir}
extract_codec_job_ids=()
langs=(en es nl de fr)
codec_choices=(English_SoundStream English_EnCodec Multi_SoundStream Multi_EnCodec )
codec_choices=( Multi_SoundStream )
for codec_choice in "${codec_choices[@]}";
do
    for lang in "${langs[@]}"
    do
    log "Start extract codec ${codec_choice} ${lang}"
    sbatch  \
            --job-name=${tag}_${codec_choice}_${lang} \
            --output=${log_dir}/${tag}_${codec_choice}_${lang}.txt \
            --error=${log_dir}/${tag}_${codec_choice}_${lang}.err \
            --time=08:00:00 \
            --partition=${gpu_cluster} \
            --qos=high_priority \
            --gpus=v100-32:1 \
            --cpus-per-task=5 \
            --mem=20G \
            --wrap="bash run.sh ${lang} ${stage} ${stop_stage} ${codec_choice} ${ngpu}"
    log "Finish extract codec ${codec_choice} ${lang}"
    done
done




