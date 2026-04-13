# JVR-Verilog-LLM

## Project Overview

This repository introduces an efficient Verilog generation framework designed to enhance Large Language Models via Joint Verification Rewards (JVR) and Dynamic Switching Mechanisms. It aims to address the limitations of current CoT-dependent models, specifically their restricted accuracy in direct (non-thinking) generation and low inference efficiency.

## Key Highlights

- **Joint Verification Reward (JVR):** A deep reinforcement learning algorithm based on **DAPO** (Decoupled Clip and Dynamic sAmpling Policy Optimization) that assigns additional rewards to samples that pass functional verification in both thinking and non-thinking modes. This implements an implicit knowledge distillation of reasoning capabilities into the fast inference path, effectively improving the quality of code generation in the model's non-thinking mode.
- **Dynamic Switching Mechanism:** An automatic model switching strategy based on output token perplexity. By monitoring preliminary generation results in real-time, the model utilizes perplexity metrics to automatically evaluate logical complexity and dynamically trigger thinking mode switching, achieving an adaptive balance between generation efficiency and hardware logic complexity.

## Evaluation Results

<p align="center">
  <img src="pics/image.png" alt="Framework Architecture" width="85%">
  <br>
  <b>Table 1:</b> <i>Performance by Training Stage on VE-V1 (%)</i>
</p>

<p align="center">
  <img src="pics/image2.png" alt="Framework Architecture" width="85%">
  <br>
  <b>Table 2:</b> <i>Performance by Training Stage on VE-V2-CC (%)</i>
</p>

<p align="center">
  <img src="pics/image3.png" alt="Framework Architecture" width="85%">
  <br>
  <b>Figure 1:</b> <i>Single-Sample Latency Comparison (s)</i>
</p>

## Quick start

1. JVR Training
To reproduce the Joint Verification Reward (JVR) training process on a dual-GPU setup:
```bash
bash scripts/mydesign_8b_2gpu.sh
```
2. Model Inference
- **Adaptive Inference:** Uses the perplexity-based switching mechanism to balance quality and speed.
```bash
bash scripts/infer_switch.sh
```
- **Original Inference:** Standard inference supporting both pure CoT and non-CoT generation.
```bash
bash scripts/infer_qwen.sh
```

## Requirements

For SFT, this project requires ms-swift 3.12.3. Please follow the github repository to install it: https://swift.readthedocs.io/en/latest/GetStarted/SWIFT-installation.html

This project requires VERL (0.8.0.dev0), a RL framework developed by bitdance. Since it cannot be installed simply via pip, please follow the github repository to install it: https://github.com/volcengine/verl

For Ubuntu 20.04, there might exists some compatibility issues, e.g., version `GLIBC_2.32` not found. In this case, please use Ubuntu 22.04 or see the following link to upgrade glibc: https://github.com/Dao-AILab/flash-attention/issues/1762, https://github.com/modular/modular/issues/3684#issuecomment-2480409734.

To install transformer-engine: SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && echo $SITE_PACKAGES && \
CUDNN_PATH=$SITE_PACKAGES/nvidia/cudnn CPLUS_INCLUDE_PATH=$SITE_PACKAGES/nvidia/cudnn/include \
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

To install flash_attn, use this command to save time: MAX_JOBS=64 python -m pip -v install flash-attn --no-build-isolation. This will cost about 500GB memory.

Or you can use official docker:

```docker
sudo docker create --runtime=nvidia --gpus all --net=host \
    --shm-size="20g" \
    --cap-add=SYS_ADMIN \
    -v $PWD:/project \
    --name verl \
    verlai/verl:vllm012.latest sleep infinity
docker start verl
docker exec -it verl bash
```

## Acknowledgments

This project is built upon the foundational work of several outstanding open-source repositories. We would like to express our sincere gratitude to the authors of the following projects:

* **[VERL](https://github.com/volcengine/verl):** Our training pipeline, particularly the custom DAPO implementation and token-level gradient updates, is heavily based on this highly efficient reinforcement learning framework.
* **[Qming-CodeV](https://github.com/iprc-dip/CodeV-R1):** We utilized their meticulously curated Verilog RL datasets, which significantly facilitated our supervised fine-tuning and reinforcement learning processes.

We sincerely thank the researchers and developers for their valuable contributions to the open-source community.
