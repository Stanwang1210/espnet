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

lang=$1
datadir=$2
train_set=$3
nutt=$(<"${datadir}/${train_set}"/wav.scp wc -l)
if [ ${lang} == "en" ]; then
    portion=0.022
elif [ ${lang} == "es" ]; then
    portion=0.99
elif [ ${lang} == "fr" ]; then
    portion=0.85
elif [ ${lang} == "nl" ]; then
    portion=0.6
elif [ ${lang} == "de" ]; then
    portion=0.5
else
    portion=1
fi
_dsets="${train_set}_subset"
log "Start subsampling ${lang} ${portion}"
portion_nutt=$(echo ${nutt} ${portion} | awk '{print(int($1 * $2))}')
portion_nutt=$(( portion_nutt > 0 ? portion_nutt : 1 ))
log "Subsampling ${portion_nutt}/${nutt} utterances from ${train_set}, saved to ${_dsets}"
utils/subset_data_dir.sh \
        "${datadir}/${train_set}" ${portion_nutt} "${datadir}/${_dsets}"
utils/fix_data_dir.sh ${datadir}/${_dsets}
utils/utt2spk_to_spk2utt.pl ${datadir}/${_dsets}/utt2spk > ${datadir}/${_dsets}/spk2utt
echo "Subsampled ${lang} ${portion} ${portion_nutt}/${nutt} utterances" > ${datadir}/${_dsets}/portion
# python3 calculate_duration.py --data_dir "${datadir}/${_dsets}"