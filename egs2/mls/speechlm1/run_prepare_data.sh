set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


log "Please set the ngpu here"
ngpu=4
langs=(es en fr nl it pt pl de)

for lang in "${langs[@]}"
do
    stage=1
    stop_stage=1
    log "Start processing data ${lang}"
    bash run.sh ${lang} ${stage} ${stop_stage}
    log "Finish processing data ${lang}"

done
log "Start combining data"
for split in "train" "dev" "test";
do
    combine_lang=""
    for lang in "${langs[@]}"
    do
        combine_lang+="data/mls_$(echo ${lang} | tr - _)_${split} " 
    done
    utils/combine_data.sh --extra_files "spk2utt"  "data/mls_all_${split}" ${combine_lang}
    echo "${langs[@]}" > "data/mls_all_${split}/all_langs.txt"
done
log "Finish combining data"

stage=2
stop_stage=2
lang="all"
log "Start processing audio data "

bash run.sh ${lang} ${stage} ${stop_stage}
log "Finish processing audio data "

stage=5
stop_stage=5
lang="all"

# codec_choices=(SoundStream EnCodec SoundStream_amuse "EnCodec_amuse")
codec_choices=(SoundStream)
for codec_choice in "${codec_choices[@]}";
do
    log "Start extract codec ${codec_choice}"
    bash run.sh ${lang} ${stage} ${stop_stage} ${codec_choice} ${ngpu}
    log "Finish extract codec ${codec_choice}"
done




