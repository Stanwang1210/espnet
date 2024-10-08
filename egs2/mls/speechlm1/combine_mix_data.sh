

train_src="dump_16000/audio_raw_tts_mls/mls_en_train_subset dump_16000/audio_raw_tts_emilia/emilia_en_train_short dump_16000/audio_raw_tts_emilia/emilia_de_train dump_16000/audio_raw_tts_mls/mls_de_train dump_16000/audio_raw_tts_emilia/emilia_fr_train dump_16000/audio_raw_tts_mls/mls_fr_train"
dev_src="dump_16000/audio_raw_tts_mls/mls_de_dev dump_16000/audio_raw_tts_mls/mls_fr_dev dump_16000/audio_raw_tts_mls/mls_en_dev"


train_dest=dump_16000/audio_raw_tts_mls/mix_train
dev_dest=dump_16000/audio_raw_bpe_tts_mls/mix_dev
# bash utils/combine_data.sh ${train_dest} ${train_src}
bash utils/combine_data.sh ${dev_dest} ${dev_src}
