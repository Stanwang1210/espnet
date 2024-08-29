set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "Please run the 'run_prepare_data.sh' script before running this script"
ngpu=2
stage=6
stop_stage=15
lang="all"

# codec_choices=(SoundStream EnCodec SoundStream_amuse "EnCodec_amuse")
codec_choices=(SoundStream)
for codec_choice in "${codec_choices[@]}";
do
    log "Start extract codec ${codec_choice}"
    bash run.sh ${lang} ${stage} ${stop_stage} ${codec_choice} ${ngpu}
    log "Finish extract codec ${codec_choice}"
done