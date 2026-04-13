python src/infer/gen_prefix.py \
    --cuda_device "1" \
    --input_file "data/inputs/hint_cc.jsonl" \
    --diff_file "data/v2_cc_outputs/cc_think_diff.json" \
    --output_file "data/v2_cc_outputs/cc_perplexity_results.jsonl" \
    --model_name "ckpts/DAPO/DAPO-Qwen3-8B-Mydesign-fixed-8roll/model_8roll_fixed_step_1200" \
    --max_tokens 16384\
    --do_sample \
    --n 1 \
    # --task_kind "nothink_passed"

    # --output_file "data/outputs/model_8roll_fixed_step_1200.jsonl" \