# vLLM

This repo is a fork of the [vLLM](https://github.com/vllm-project/vllm) repo.

## Usage

Pull the `latest` image from ECR:

```bash
bash docker/pull.sh vllm:latest
```

Run the container (with Llama3 8B in this case):

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

## Development

### Setup dev mode

Clone the repo and setup the base docker image:

```bash
docker run --gpus all -it --rm --ipc=host \
	-v $(pwd):/workspace/vllm \
	-v ~/.cache/huggingface:/root/.cache/huggingface \
	-p 8000:8000 \
	nvcr.io/nvidia/pytorch:23.10-py3
```

Once done, install vLLM in dev mode and the dev requirements in the container:

```bash
cd vllm
pip install -e .
pip install -r requirements-dev.txt
pip install boto3
```

It will take a while but once done, open another terminal **on the host** and run:

```bash
docker commit <container_id> vllm_dev
```

This will create a new image `vllm_dev` with the vLLM code installed. You won't need to install the dev dependencies again each time you start a new container.

From now on, you can exit the initial container and run this command to enter into the dev container:

```bash
docker run --gpus all -it --rm --ipc=host \
	-v $(pwd):/workspace/vllm \
	-v ~/.cache/huggingface:/root/.cache/huggingface \
	-p 8000:8000 \
	vllm_dev
```

### Launch the server

Enter into the `vllm_dev` container and run:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

### Format the code

Enter into the `vllm_dev` container and run:

```bash
bash format.sh
```

### Build the image

Once your changes are ready, you can build the prod image. Run these commands **on the host**:

```bash
bash docker/build.sh
```

And deploy it to ECR:

```bash
bash docker/deploy.sh <version>
```

### Upgrade version

You can upgrade the version of vLLM by rebasing on the official repo:

```bash
git clone https://github.com/lightonai/vllm
git remote add official https://github.com/vllm-project/vllm
git fetch official
git rebase <commit_sha> # Rebase on a specific commit of the official repo (i.e. the commit sha of the last stable release)
git rebase --continue # After resolving conflicts (if any), continue the rebase
git push origin main --force
```

## Deployment

To deploy a model on Sagemaker, follow this [README](https://github.com/lightonai/vllm/blob/main/sagemaker/README.md).
