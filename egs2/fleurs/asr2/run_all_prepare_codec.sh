#!/bin/bash

# All languages
langs=("ceb_ph" "fil_ph" "gu_in" "jv_id" "kea_cv" "kam_ke" "kn_in" "lo_la" "ln_cd" "luo_ke" "mi_nz" "ny_mw" "sn_zw" "sd_in" "umb_ao" "wo_sn")
current_date=$(date +%Y-%m-%d)
task=asr
tag=run_prepare_${task}_codec

codec_choice=ESPnet

if [ ${codec_choice} == "ESPnet" ]; then
    fs=16000
    codec_opts="--codec_choice ${codec_choice} --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"
elif [ ${codec_choice} == "EnCodec" ]; then
    fs=24000
    codec_opts="--codec_choice ${codec_choice}"
else
    echo "Unknown codec choice"
    exit 1
fi

log_dir=slurm_logs/${current_date}/${tag}_${codec_choice}_${fs}
mkdir -p ${log_dir}
# rm -r ${log_dir}/*.txt
# rm -r ${log_dir}/*.err

# For slurm
for lang in "${langs[@]}"
do
  sbatch --job-name=prepare_data_${lang} \
         --output=${log_dir}/prepare_data_${lang}_%j.txt \
         --error=${log_dir}/prepare_data_${lang}_%j.err \
         --time=2-00:00:00 \
         --partition=RM-shared \
         --cpus-per-task=9 \
         --mem=16G \
         --wrap="bash run_prepare_codec.sh ${lang} ${fs} '${codec_opts}' "
done

# For battleship
# for lang in "${langs[@]}"
# do
#   hrun -c 8 -m 50 -G -g 3090 -t 7-0 \
#   --image /livingrooms/public/singularity‐images/default_20240430.sif \
#   "bash run_prepare_codec.sh ${lang} ${fs} '${codec_opts}' "
# done