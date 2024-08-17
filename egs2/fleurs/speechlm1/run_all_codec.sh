#!/bin/bash

# 語言列表
current_date=$(date +%Y-%m-%d)
langs=("ceb_ph" "fil_ph" "gu_in" "jv_id" "kea_cv" "kam_ke" "kn_in" "lo_la" "ln_cd" "luo_ke" "mi_nz" "ny_mw" "sn_zw" "sd_in" "umb_ao" "wo_sn")
# langs=("ceb_ph")
# 提交每個語言的作業

task=tts
codec_choice=ESPnet
tag=run_codec_unseen_codec_${task}_${codec_choice}


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
# codec_opts="--codec_choice ${codec_choice} "
asr_config=conf/tuning/train_asr_ebranchformer_${codec_choice}.yaml
log_dir=slurm_logs/${current_date}/${tag}_${codec_choice}_${fs}
expdir=exp_${task}
mkdir -p ${log_dir}
# rm -r ${log_dir}/*.txt
# rm -r ${log_dir}/*.err
# for lang in "${undone_langs[@]}"
# do
#   sbatch --job-name=${tag}_${lang} \
#          --output=${log_dir}/${tag}_${lang}_%j.txt \
#          --error=${log_dir}/${tag}_${lang}_%j.err \
#          --time=2-00:00:00 \
#          --partition=RM-shared \
#          --cpus-per-task=9 \
#          --mem=17G \
#          --wrap="bash run_codec_unseen.sh ${lang}"
# done
for lang in "${langs[@]}"
do
  sbatch --job-name=${tag}_${lang} \
         --output=${log_dir}/${tag}_${lang}.txt \
         --error=${log_dir}/${tag}_${lang}.err \
         --time=2-00:00:00 \
         --partition=GPU-shared \
         --cpus-per-task=4 \
         --gpus=v100-32:1 \
         --mem=15G \
         --wrap="bash run_codec.sh ${lang} ${fs} ${asr_config} '${codec_opts}' ${expdir}"
done

# For battleship
for lang in "${langs[@]}"
do
  hrun -c 8 -m 50 -G -g 3090 -t 7-0 \
  --image /livingrooms/public/singularity‐images/default_20240430.sif \
  "bash run_codec.sh ${lang} ${fs} ${asr_config} '${codec_opts} ${expdir}"
done