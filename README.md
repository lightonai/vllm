# vLLM

This repo is a fork of the [vLLM](https://github.com/vllm-project/vllm) repo.

## Usage

Pull the `latest` image from ECR:

```bash
bash docker/pull.sh vllm:latest
```

Run the container (with Command-R model in this case):

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    -e SERVED_MODEL_NAME=command-r \
    -e TRUST_REMOTE_CODE=false \
    -e MODEL=CohereForAI/c4ai-command-r-v01 \
    vllm \
    --tensor-parallel-size 4 \
    --host 0.0.0.0
```

## Development

### Build the image

```bash
sh docker/build.sh
```

### Deploy the image to ECR

Once your changes are ready, you can deploy the image to ECR:

```bash
sh docker/deploy.sh
```

### Upgrade version

You can upgrade the version of vLLM by rebasing on the official repo:

```bash
git clone https://github.com/lightonai/vllm
git remote add official https://github.com/vllm-project/vllm
git fetch official
git rebase official/main
git rebase --continue # After resolving conflicts (if any), continue the rebase
git push origin main --force
```

## Deployment

To deploy a model on Sagemaker, follow this [README](https://github.com/lightonai/vllm/blob/main/sagemaker/README.md).
