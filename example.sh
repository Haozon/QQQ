model_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/LLMs/LLMs_HF/Qwen3-30B-A3B
tokenizer_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/LLMs/LLMs_HF/Qwen3-30B-A3B
save_path=./output/Qwen3-30B-A3B-Quant

## per-channel wegiht quantization
python3 examples/quant_model.py \
    --model_path ${model_path} \
    --tokenizer_path ${tokenizer_path} \
    --dtype float16 \
    --smooth false \
    --rotation true \
    --dataset wikitext2 \
    --nsamples 128 \
    --w_quantizer FixedQuantize \
    --w_group_size -1 \
    --gptq_mse true \
    --gptq_groupsize -1 \
    --save_path ${save_path} \
    