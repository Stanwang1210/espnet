set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
# shellcheck disable=SC1091
. ./path.sh || exit 1;
# shellcheck disable=SC1091
. ./cmd.sh || exit 1;
# shellcheck disable=SC1091
. ./db.sh || exit 1;
declare -A g2p_dict=(
    ["de"]="espeak_ng_german"
    ["el"]="espeak_ng_greek"
    ["es"]="espeak_ng_spanish"
    ["fi"]="espeak_ng_finnish"
    ["fr"]="espeak_ng_french"
    ["hu"]="espeak_ng_hungarian"
    ["ja"]="pyopenjtalk"
    ["nl"]="espeak_ng_dutch"
    ["ru"]="espeak_ng_russian"
    ["zh"]="pypinyin_g2p_phone"
)
lang=$1
nj=16
threshold=35
python process_commonvoice.py ${lang}

log "scripts/audio/trim_silence.sh"
# shellcheck disable=SC2154
scripts/audio/trim_silence.sh \
    --cmd "${train_cmd}" \
    --nj "${nj}" \
    --fs 16000 \
    --win_length 1024 \
    --shift_length 256 \
    --threshold "${threshold}" \
    "data/commonvoice_${lang}_test" "data/commonvoice_${lang}_test/log"

log " pyscripts/utils/convert_text_to_phn.py"
g2p=${g2p_dict[${lang}]}
utils/copy_data_dir.sh "data/commonvoice_${lang}_test" "data/commonvoice_${lang}_test_phn"
pyscripts/utils/convert_text_to_phn.py \
    --g2p "${g2p}" --nj "${nj}" \
    "data/commonvoice_${lang}_test/text" "data/commonvoice_${lang}_test_phn/text"
utils/fix_data_dir.sh "data/commonvoice_${lang}_test_phn"