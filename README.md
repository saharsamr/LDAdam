# Low-Dimensional Adam

This repository contains a reference torch implementation of the [LDAdam optimizer](./ldadamw_torch/ldadamw.py) as proposed in the paper: [LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics]().

**Abstract:** We introduce LDAdam, a memory-efficient optimizer for training large models, that performs adaptive optimization steps within lower dimensional subspaces, while consistently exploring the full parameter space during training. This strategy keeps the optimizer's memory footprint to a fraction of the model size. LDAdam relies on a new projection-aware update rule for the optimizer states that allows for transitioning between subspaces, i.e., estimation of the statistics of the projected gradients. To mitigate the errors due to low-rank projection, LDAdam integrates a new generalized error feedback mechanism, which explicitly accounts for both gradient and optimizer state compression. We prove the convergence of LDAdam under standard assumptions, and show that LDAdam allows for accurate and efficient fine-tuning and pre-training of language models.

## LDAdam Optimizer Usage

To integrate the optimizer into your own pipeline, please use the following snippet:

```python
from LowDimensionalAdam import LDAdamW

# define param groups as fullrank_params and lowrank_params
optimizer = LDAdamW(
    params=[{'params': fullrank_params, 'enable_lowrank': False},
    {'params': lowrank_params, 'enable_lowrank': True, 'rank':16, 'rho':0.908}],
    lr=0.001,
    betas=(0.908,0.99),
    eps=1e-8,
    weight_decay=0.0,
)

# you can then use the variable `optimizer` as any other PyTorch optimizer
```

## Installation

To install LDAdam latest stable version from source, please run:

```shell
pip3 install git+https://github.com/IST-DASLab/LDAdam.git
```

To clone the project and install it as a Python package in a new conda environment named LDAdam, please run:

```shell
git clone https://github.com/IST-DASLab/LDAdam.git
cd LDAdam
source install.sh
conda activate LDAdam
```

## Reproduce Experiments

### Fine-tuning BERT Model for GLUE Benchmark

To conduct experiments on fine-tuning RoBERTa-base model on the [GLUE benchmark](https://github.com/md-experiments/glue_benchmark?tab=readme-ov-file), we rely on the [Huggingface Transformers project](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification). The additional dependencies required can be found in the file [glue_requirements.txt](./experiments/glue_finetuning/glue_requirements.txt). To install them, please run:

```shell
pip3 install -r experiments/glue_finetuning/glue_requirements.txt
```

For reproductibility purposes, we provide the [scripts](./experiments/glue_finetuning/scripts) we used to run our experiments.

### Pre-training Llama Model on the C4 dataset

To conduct our experiments on Llama pre-training on the C4 dataset, we follow the training procedure provided by the [ReLora project](https://github.com/Guitaricet/relora) and adapted for the [GaLore project](https://github.com/jiaweizzhao/GaLore). The additional dependencies required can be found in the file [c4_requirements.txt](./experiments/c4_pretraining/c4_requirements.txt). To install them, please run:

```shell
pip3 install -r experiments/c4_pretraining/c4_requirements.txt
```

For reproductibility purposes, we provide the [scripts](./experiments/c4_pretraining/scripts) we used to run our experiments.

### Finetuning Llama Model on the GSM8K dataset

To run our experiments on fine-tuning the Llama2 7B model on the [GSM8K dataset](https://github.com/openai/grade-school-math), we use the training pipeline provided by the [MicroAdam project](https://github.com/IST-DASLab/MicroAdam/), based on [MosaicML's LLM foundry framework](https://github.com/mosaicml/llm-foundry).

## Citation