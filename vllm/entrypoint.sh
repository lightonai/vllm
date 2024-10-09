#!/bin/bash

# Check if the first argument is 'serve'
if [ "$1" = "serve" ]; then
    # Construct the command with inline conditionals
    CMD=("python3" "-m" "vllm.entrypoints.openai.api_server" "--host" "0.0.0.0" "--port" "8080" "--download-dir" "/tmp/models"
        ${MODEL:+--model $MODEL}
        ${SERVED_MODEL_NAME:+--served-model-name $SERVED_MODEL_NAME}
        ${TRUST_REMOTE_CODE:+--trust-remote-code}
        ${MAX_MODEL_LEN:+--max-model-len $MAX_MODEL_LEN}
        ${PIPELINE_PARALLEL_SIZE:+--pipeline-parallel-size $PIPELINE_PARALLEL_SIZE}
        ${TENSOR_PARALLEL_SIZE:+--tensor-parallel-size $TENSOR_PARALLEL_SIZE}
        ${MAX_NUM_SEQS:+--max-num-seqs $MAX_NUM_SEQS}
        ${DISABLE_CUSTOM_ALL_REDUCE:+--disable-custom-all-reduce}
        ${ENABLE_LORA:+--enable-lora}
        ${MAX_LORAS:+--max-loras $MAX_LORAS}
        ${MAX_LORA_RANK:+--max-lora-rank $MAX_LORA_RANK}
        ${LORA_MODULES:+--lora-modules $LORA_MODULES}
    )
    echo "Running command: ${CMD[@]}"
    exec "${CMD[@]}"
else
    # If the argument is not 'serve', pass all arguments to the original entrypoint
    exec python3 -m vllm.entrypoints.openai.api_server "$@"
fi
