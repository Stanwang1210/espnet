current_date=$(date +%Y-%m-%d)
# langs=(es en fr nl de)
conda_root=/ocean/projects/cis210027p/swang26/miniconda3
env_name=espnet_codec_robustness
tag=upload_data
cpu_cluster=RM-shared
log_dir=slurm_logs/${current_date}/${tag}

mkdir -p ${log_dir}
langs=(es )
for lang in ${langs[@]}; do
        sbatch  --job-name=${tag}_${lang} \
                --output=${log_dir}/${tag}_${lang}.txt \
                --error=${log_dir}/${tag}_${lang}.err \
                --time=12:00:00 \
                --partition=${cpu_cluster} \
                --qos=high_priority \
                --cpus-per-task=8 \
                --mem=15G \
                --wrap="source ${conda_root}/envs/${env_name}/etc/profile.d/conda.sh; conda activate ${conda_root}/envs/${env_name}; huggingface-cli upload --repo-type dataset audio_raw_tts_mls_${lang} audio_raw_tts_mls/mls_${lang}_train_subset"
done