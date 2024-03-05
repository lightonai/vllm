#!/bin/bash

# Check if the first argument is 'serve'
if [ "$1" = "serve" ]; then
    exec python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8080 --trust-remote-code --download-dir="/tmp/models"
else
    # If the argument is not 'serve', pass all arguments to the original entrypoint
    exec python3 -m vllm.entrypoints.openai.api_server "$@"
fi
