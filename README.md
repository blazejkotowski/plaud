# Performative Latents for Adaptive Unsupervised DDSP (PLAUD)

**PLAUD** is a PyTorch-based modular synthesis framework that extends [DDSP: Differentiable Digital Signal Processing](https://arxiv.org/abs/2001.04643) for **real-time**, **performance-oriented**, and **latent-variable-controlled** sound generation. It is specifically designed for **small, personal datasets**, and prioritizes **playability**, **modularity**, and **exploration** over strict reconstruction quality.

At its core, PLAUD is a reconfigurable DDSP synthesizer where **every aspect of the synthesis and training process is steered by a structured latent space**. It supports **intervenable architectures**, allowing real-time constraints (e.g., number of oscillators) and flexible loss objectives. 

## Key Features

- **Exclusively latent-based control**: The entire generation process is conditioned on a regularized latent space—there is no explicit audio feature extraction.
- **Smoothed latent trajectories**: Temporal structure and controllability are improved via latent smoothing (e.g., average pooling over time).
- **Modular synthesis blocks**:
  - **Sinusoidal** additive synthesis
  - **Harmonic** oscillator banks (optional)
  - **NoiseBandNet**-based residual modeling ([NoiseBandNet paper](https://arxiv.org/abs/2307.08007))
- **Customizable loss functions**:
  - Multi-resolution STFT loss
  - Perceptual **CLAP** loss (audio-text contrastive model)
  - Perceptual **M2L** loss (mel-to-latent for realism)
- **Optional attribute regularization**: Add task-specific structure to latent space for controlled generation. ([Attribute Regularization paper](https://arxiv.org/abs/2004.05485)
- **Highly customizable**: Modular training interface for swapping losses, synthesis blocks, and regularization strategies.
- **Real-time deployment**: Compatible with `nn~` externals for **Max/MSP** and **PureData** for live performance environments.

---

## Installation
Clone the repository and install the package locally with pip:
```bash
pip install -r requirements.txt
pip install -e .
```

## Training
The training is done in two steps:
1. Preprocess the dataset
```bash
python utils/dataset_converter.py --input_dir <path_to_dataset> --output_dir <path_to_output_dir>
```
2. Train the model
```bash
python cli/train.py\
        --latent_size 8\
        --model_name <model_name>\
        --dataset_path <path_to_preprocessed_dataset>
```

The training process is highly customisable. To see all the options run:
```bash
python cli/train.py --help
```
## Inference

### Max/MSP and PureData
The model is compatibile with nn~ externals for Max/MSP and PureData. In order to use trained model, you need to install the extensions following the instructions from [original nn~ repository](https://github.com/acids-ircam/nn_tilde).

### Model export
In order to export the model to be used with nn~ externals, run:
```bash
python cli/export.py --model_directory <path_to_model_training> --output_dir <path_to_output_dir>
```


## Colab Notebook
The project is also available as this [colab notebook](https://colab.research.google.com/drive/1otNApPfqy9DcbyX1Jaxu1Xc1AR9RrEzT#scrollTo=ABldwFMwll7j).
