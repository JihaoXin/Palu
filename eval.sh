#!/bin/bash -l
#SBATCH --job-name=palu_eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --mem=100G
#SBATCH --time=2:00:00

for RATIO in 0.5 0.6 0.7 0.8 0.9; do
  python compress.py --model_id=meta-llama/Meta-Llama-3-8B-Instruct --calib_dataset wikitext2 --param_ratio_target ${RATIO} --search_method fisher_uniform --head_group_size 4 --dump_huggingface_model --use_cache --decompose_method=svd
  python -u run_ppl_eval.py --model_name_or_path Meta-Llama-3-8B-Instruct_ratio-${RATIO}_gs-4-fisher_uniform-svd --datasets wikitext2 --seqlen 2048 > /home/xinj/rap/logs/svd_kv_${RATIO}_ppl.log 2>&1
  python -u run_lm_eval.py --model_name_or_path=Meta-Llama-3-8B-Instruct_ratio-${RATIO}_gs-4-fisher_uniform-svd --tasks="openbookqa" > /home/xinj/rap/logs/svd_kv_${RATIO}_accuracy.log 2>&1
done
