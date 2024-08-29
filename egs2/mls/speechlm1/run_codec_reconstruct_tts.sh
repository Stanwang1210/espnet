set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


log "Please run the 'run_prepare_data.sh' script before running this script"
ngpu=2
stage=3
stop_stage=3

# codec_choices=(SoundStream EnCodec SoundStream_amuse "EnCodec_amuse")
codec_choices=(SoundStream)
for codec_choice in "${codec_choices[@]}";
do 
    bash run_nar_tts.sh ${codec_choice} ${stage} ${stop_stage} ${ngpu}
done

stage=4
stop_stage=15
for codec_choice in "${codec_choices[@]}";
do 
    bash run_nar_tts.sh ${codec_choice} ${stage} ${stop_stage} ${ngpu}
done