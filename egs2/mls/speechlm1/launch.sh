set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
conda_root=/ocean/projects/cis210027p/swang26/miniconda3
env_name=espnet_codec
source ${conda_root}/envs/${env_name}/etc/profile.d/conda.sh
conda activate ${conda_root}/envs/${env_name}

conda_cmd="conda_root=/ocean/projects/cis210027p/swang26/miniconda3;env_name=espnet_codec;source ${conda_root}/envs/${env_name}/etc/profile.d/conda.sh;conda activate ${conda_root}/envs/${env_name};"

current_date=$(date +%Y-%m-%d)

lang=$1
ngpu=$2
cmd="bash run.sh ${lang} ${ngpu}"
tag=tts_train_${lang}
log_dir=slurm_logs/${current_date}/${tag}
mkdir -p ${log_dir}
sbatch  \
    --dependency=afterok:26076071 \
    --job-name=${tag} \
    --output=${log_dir}/${tag}.txt \
    --error=${log_dir}/${tag}.err \
    --time=08:00:00 \
    --partition=GPU-small \
    --qos=high_priority \
    --gpus=v100-32:${ngpu} \
    --cpus-per-task=10 \
    --mem=30G \
    --wrap="${conda_cmd} ${cmd}"
log "Submitted job: ${cmd}"
# sbatch  \
#     --job-name=${tag} \
#     --output=${log_dir}/${tag}.txt \
#     --error=${log_dir}/${tag}.err \
#     --time=03:00:00 \
#     --partition=RM-shared \
#     --qos=high_priority \
#     --cpus-per-task=64 \
#     --mem=120G \
#     --wrap="${conda_cmd} ${cmd}"