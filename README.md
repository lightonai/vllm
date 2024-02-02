# vLLM

This repo is a fork of the [vLLM](https://github.com/vllm-project/vllm) repo.

## Build the image

```bash
sh docker/build.sh
```

## Deploy the image to ECR

```bash
sh docker/deploy.sh
```

## Upgrade version

You can upgrade the version of vLLM by rebasing on the official repo:

```bash
git clone https://github.com/lightonai/vllm
git remote add official https://github.com/vllm-project/vllm
git fetch official
git rebase official/main
git rebase --continue # After resolving conflicts (if any), continue the rebase
git push origin main --force
```
