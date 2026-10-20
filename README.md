# MNIST Diffusion Model using PyTorch

A deep learning project that implements a **Denoising Diffusion Probabilistic Model (DDPM)** to generate handwritten digit images using the MNIST dataset. The model learns to reverse a gradual noise-addition process using a time-conditioned U-Net architecture.

## Project Overview

Diffusion models are generative models that learn to create new data by progressively removing noise from randomly generated inputs.

In this project, a neural network is trained on handwritten digits from MNIST. During training, Gaussian noise is added to images at different timesteps, and the model learns to predict that noise. During inference, the trained model iteratively denoises random noise to generate new digit-like images.

### Key Features

- Implementation of a diffusion model using PyTorch.
- MNIST handwritten digit dataset.
- Time-conditioned U-Net architecture.
- Forward diffusion and reverse denoising processes.
- GPU acceleration using CUDA when available.
- Iterative image generation and visualization.
- Model checkpoint saving for future inference.

## Model Configuration

| Parameter | Value |
|---|---|
| Dataset | MNIST |
| Image resolution | 32 × 32 |
| Image channels | 1 (grayscale) |
| Diffusion timesteps | 300 |
| Noise schedule | Linear |
| Starting beta | 0.0001 |
| Ending beta | 0.02 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Training epochs | 20 |
| Optimizer | Adam |
| Loss function | L1 loss |
| Framework | PyTorch |

*Note: These settings describe the notebook configuration. Actual training results depend on execution and model convergence.*

## Technologies Used

- **Python** — Programming language
- **PyTorch** — Deep learning framework
- **Torchvision** — Dataset loading and image transformations
- **NumPy** — Numerical operations
- **Matplotlib** — Image visualization
- **tqdm** — Training progress monitoring
- **Google Colab** — GPU-enabled development environment

## How It Works

### 1. Forward Diffusion

Gaussian noise is progressively added to a clean MNIST image over a sequence of timesteps. At larger timesteps, the image becomes increasingly noisy.

### 2. Noise Prediction

A time-conditioned U-Net receives a noisy image and its diffusion timestep. It learns to predict the noise added to the image during the forward process.

The model is trained by comparing the predicted noise with the actual sampled noise using the configured L1 loss.

### 3. Reverse Diffusion

Generation starts with random Gaussian noise. The trained model predicts the noise at each timestep, and the reverse diffusion process gradually removes it to produce a new image.

## Installation and Usage

### Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `diffusion-model.ipynb`.
3. Navigate to **Runtime → Change runtime type**.
4. Select a GPU runtime if available.
5. Execute the notebook cells in order.

The notebook downloads the MNIST dataset as needed.

### Run Locally

Clone the repository:

```bash
git clone https://github.com/Abhinav001-glitch/diffusion-mnist.git
cd diffusion-mnist
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook diffusion-model.ipynb
```

## Expected Outputs

After successfully executing the notebook, you should be able to inspect:

- Generated samples of handwritten digit-like images.
- Sample grids saved during training, if enabled in the notebook.
- The trained model checkpoint, if checkpoint saving completes successfully.

The notebook is configured to use the following output paths:

```text
diffusion_outputs/
├── epoch_1_sample.png
├── epoch_2_sample.png
├── ...
├── epoch_20_sample.png
└── diffusion_mnist.pth
```

These are expected output paths, not files included in the repository by default. Check the notebook for the exact save operations.

## Hardware Requirements

- Python 3.10 or newer is recommended.
- A CUDA-compatible GPU is recommended for training.
- CPU execution may be significantly slower.
- GPU memory requirements depend on the U-Net architecture and batch size.

If you encounter CUDA out-of-memory errors, consider reducing the batch size or model channel widths.

## Learning Outcomes

This project provides practical experience with:

- Generative deep learning.
- Denoising Diffusion Probabilistic Models (DDPMs).
- U-Net architectures.
- Time-step embeddings.
- Gaussian noise schedules.
- Training neural networks with PyTorch.
- GPU-based model training.
- Iterative image generation.

## Future Improvements

- Add training and validation loss plots.
- Compare different noise schedules.
- Experiment with MSE versus L1 noise-prediction losses.
- Optimize the U-Net for lower-memory GPUs.
- Add reproducible random seeds and configurable hyperparameters.
- Implement a separate inference script.
- Evaluate generated images using suitable generative-model metrics.
- Add sample images and a demonstration GIF to the README.

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.

## Author

**Abhinav Kumar**

- GitHub: [Abhinav001-glitch](https://github.com/Abhinav001-glitch)
- Kaggle: [abhinav4651](https://www.kaggle.com/abhinav4651)

---

*Built as a hands-on deep learning project to explore diffusion models and image generation using PyTorch.*
