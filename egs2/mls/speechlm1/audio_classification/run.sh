
codec_choice="Multi_Soundstream"

if [ ${codec_choice} == "English_EnCodec" ]; then
    config_file=conf/train_esc_multi_soundstream.yaml
    model_tag="espnet/mls-english_encodec_16k_360epoch"
elif [ ${codec_choice} == "Multi_EnCodec" ]; then
    model_tag="espnet/mls-multi_encodec_16k_360epoch"
elif [ ${codec_choice} == "Audio_EnCodec" ]; then
    model_tag="espnet/mls-audioset_encodec_16k_360epoch"
elif [ ${codec_choice} == "English_Soundstream" ]; then
    model_tag="espnet/mls-english_soundstream_16k"
elif [ ${codec_choice} == "Multi_Soundstream" ]; then
    config_file=conf/train_esc_multi_soundstream.yaml
    model_tag="espnet/mls-multi_soundstream_16k"
elif [ ${codec_choice} == "Audio_Soundstream" ]; then
    config_file=conf/train_esc_audioset_soundstream.yaml
    model_tag="espnet/mls-audioset_soundstream_16k"

else
    echo "Unknown codec choice"
    exit 1
fi

dumpdir=dump_16000/raw_tts_mlsE_SPnet__$(echo ${model_tag} | tr '/' '_')
exptag=$(basename ${config_file} .yaml)
expdir=exp_esc/${exptag}

mkdir -p ${expdir}
cp ${config_file} ${expdir}/config.yaml

python run.py \
--config_file ${config_file} \
--model_tag ${model_tag} \
--dumpdir ${dumpdir} \
--exp_dir ${expdir} \