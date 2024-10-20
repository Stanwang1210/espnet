
codec_choice=$1
gpu_id=0

if [ ${codec_choice} == "English_EnCodec" ]; then
    config_file=audio_classification/conf/train_esc_multi_soundstream.yaml
    model_tag="espnet/mls-english_encodec_16k_360epoch"
elif [ ${codec_choice} == "Multi_EnCodec" ]; then
    model_tag="espnet/mls-multi_encodec_16k_360epoch"
elif [ ${codec_choice} == "Audio_EnCodec" ]; then
    model_tag="espnet/mls-audioset_encodec_16k_360epoch"
elif [ ${codec_choice} == "English_Soundstream" ]; then
    model_tag="espnet/mls-english_soundstream_16k"
elif [ ${codec_choice} == "Multi_Soundstream" ]; then
    config_file=audio_classification/conf/train_esc_multi_soundstream.yaml
    model_tag="espnet/mls-multi_soundstream_16k"
elif [ ${codec_choice} == "Audio_Soundstream" ]; then
    config_file=audio_classification/conf/train_esc_audioset_soundstream.yaml
    model_tag="espnet/mls-audioset_soundstream_16k"

else
    echo "Unknown codec choice"
    exit 1
fi

dumpdir=dump_16000/raw_tts_esc_ESPnet_$(echo ${model_tag} | tr '/' '_')
exptag=$(basename ${config_file} .yaml)

echo ${config_file}


which python

for fold in 1 2 3 4 5; do
    expdir=exp_esc/${exptag}/fold_${fold}
    echo ${expdir}
    mkdir -p ${expdir}
    cp ${config_file} ${expdir}/config.yaml
    CUDA_VISIBLE_DEVICES="${gpu_id}" python audio_classification/run.py \
    --config_file ${config_file} \
    --model_tag ${model_tag} \
    --dumpdir ${dumpdir} \
    --exp_dir ${expdir} \
    --fold ${fold} \
    --skip_train &
done
wait