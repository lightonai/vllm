# vLLM

This repo is a fork of the [vLLM](https://github.com/vllm-project/vllm) repo.

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://vllm.ai"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://discord.gg/jz7wjKhh6g"><b>Discord</b></a> |

</p>

*Latest News* 🔥
- [2024/04] We hosted [the third vLLM meetup](https://robloxandvllmmeetup2024.splashthat.com/) with Roblox! Please find the meetup slides [here](https://docs.google.com/presentation/d/1A--47JAK4BJ39t954HyTkvtfwn0fkqtsL8NGFuslReM/edit?usp=sharing).
- [2024/01] We hosted [the second vLLM meetup](https://lu.ma/ygxbpzhl) in SF! Please find the meetup slides [here](https://docs.google.com/presentation/d/12mI2sKABnUw5RBWXDYY-HtHth4iMSNcEoQ10jDQbxgA/edit?usp=sharing).
- [2024/01] Added ROCm 6.0 support to vLLM.
- [2023/12] Added ROCm 5.7 support to vLLM.
- [2023/10] We hosted [the first vLLM meetup](https://lu.ma/first-vllm-meetup) in SF! Please find the meetup slides [here](https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s/edit?usp=sharing).
- [2023/09] We created our [Discord server](https://discord.gg/jz7wjKhh6g)! Join us to discuss vLLM and LLM serving! We will also post the latest announcements and updates there.
- [2023/09] We released our [PagedAttention paper](https://arxiv.org/abs/2309.06180) on arXiv!
- [2023/08] We would like to express our sincere gratitude to [Andreessen Horowitz](https://a16z.com/2023/08/30/supporting-the-open-source-ai-community/) (a16z) for providing a generous grant to support the open-source development and research of vLLM.
- [2023/07] Added support for LLaMA-2! You can run and serve 7B/13B/70B LLaMA-2s on vLLM with a single command!
- [2023/06] Serving vLLM On any Cloud with SkyPilot. Check out a 1-click [example](https://github.com/skypilot-org/skypilot/blob/master/llm/vllm) to start the vLLM demo, and the [blog post](https://blog.skypilot.co/serving-llm-24x-faster-on-the-cloud-with-vllm-and-skypilot/) for the story behind vLLM development on the clouds.
- [2023/06] We officially released vLLM! FastChat-vLLM integration has powered [LMSYS Vicuna and Chatbot Arena](https://chat.lmsys.org) since mid-April. Check out our [blog post](https://vllm.ai).

---
## About
vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with **PagedAttention**
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantization: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SqueezeLLM](https://arxiv.org/abs/2306.07629), FP8 KV Cache
- Optimized CUDA kernels

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support NVIDIA GPUs and AMD GPUs
- (Experimental) Prefix caching support
- (Experimental) Multi-lora support

vLLM seamlessly supports many Hugging Face models, including the following architectures:

- Aquila & Aquila2 (`BAAI/AquilaChat2-7B`, `BAAI/AquilaChat2-34B`, `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.)
- Baichuan & Baichuan2 (`baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.)
- BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
- ChatGLM (`THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, etc.)
- Command-R (`CohereForAI/c4ai-command-r-v01`, etc.)
- DBRX (`databricks/dbrx-base`, `databricks/dbrx-instruct` etc.)
- DeciLM (`Deci/DeciLM-7B`, `Deci/DeciLM-7B-instruct`, etc.)
- Falcon (`tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.)
- Gemma (`google/gemma-2b`, `google/gemma-7b`, etc.)
- GPT-2 (`gpt2`, `gpt2-xl`, etc.)
- GPT BigCode (`bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, etc.)
- GPT-J (`EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.)
- GPT-NeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.)
- InternLM (`internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.)
- InternLM2 (`internlm/internlm2-7b`, `internlm/internlm2-chat-7b`, etc.)
- Jais (`core42/jais-13b`, `core42/jais-13b-chat`, `core42/jais-30b-v3`, `core42/jais-30b-chat-v3`, etc.)
- LLaMA, Llama 2, and Meta Llama 3 (`meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
- MiniCPM (`openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, etc.)
- Mistral (`mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.)
- Mixtral (`mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc.)
- MPT (`mosaicml/mpt-7b`, `mosaicml/mpt-30b`, etc.)
- OLMo (`allenai/OLMo-1B`, `allenai/OLMo-7B`, etc.)
- OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.)
- Orion (`OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat`, etc.)
- Phi (`microsoft/phi-1_5`, `microsoft/phi-2`, etc.)
- Qwen (`Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.)
- Qwen2 (`Qwen/Qwen1.5-7B`, `Qwen/Qwen1.5-7B-Chat`, etc.)
- Qwen2MoE (`Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc.)
- StableLM(`stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2`, etc.)
- Starcoder2(`bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b`, etc.)
- Xverse (`xverse/XVERSE-7B-Chat`, `xverse/XVERSE-13B-Chat`, `xverse/XVERSE-65B-Chat`, etc.)
- Yi (`01-ai/Yi-6B`, `01-ai/Yi-34B`, etc.)

Install vLLM with pip or [from source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source):
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
