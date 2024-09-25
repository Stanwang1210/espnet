inference_dir=/mnt/data/stan/codec_espnet/egs2/mls/speechlm1/exp_ar_tts_phn/speechlm_tts_mls_en_train_valle_espnet_mls-multi_soundstream_16k/decode_espnet_multi_soundstream_valid.total_count.best
test_set=tts_mls_en_test
out_dir=eval_spk_results/${test_set}
threshold=2.7
mkdir -p ${out_dir}

echo "Collecting WER and MOS scores for ${test_set} set"
echo "Inference dir: ${inference_dir}"
echo "Output dir: ${out_dir}"
python3 eval_spk_toolkit/collect_wer_spk.py \
--u $inference_dir/${test_set}/scoring/eval_wer/utt_result.txt \
--o ${out_dir}

python3 eval_spk_toolkit/collect_mos_spk.py \
--u $inference_dir/${test_set}/scoring/eval_spk_mos/utt_result.txt \
--o ${out_dir}

python3 eval_spk_toolkit/filter_spk.py \
--m ${out_dir}/spk_mos.json \
--t ${threshold} \
--o ${out_dir}

python3 eval_spk_toolkit/collect_wer_spk.py \
--u $inference_dir/${test_set}/scoring/eval_wer/utt_result.txt \
--s ${out_dir}/spk_mos_gt_${threshold}.txt \
--o ${out_dir}



# python3 eval_spk_toolkit/combine_spk_result.py \
# --w ${out_dir}/spk_wer.json \
# --m ${out_dir}/spk_mos.json \
# --o ${out_dir}

# python3 eval_spk_toolkit/combine_utt_result.py \
# --w ${out_dir}/utt_wer.json \
# --m ${out_dir}/utt_mos.json \
# --o ${out_dir}

# python3 eval_spk_toolkit/calculate_correlation.py \
# --c ${out_dir}/wer_vs_spkmos.csv \
# --o ${out_dir}

# python3 eval_spk_toolkit/calculate_correlation.py \
# --c ${out_dir}/wer_vs_uttspkmos.csv \
# --o ${out_dir}
