# Probabilistic Center-Surround Model

Bayesian inference based normative model that explains in-vivo surround modulation results in Fu et al. 2026: *Statistics of natural scenes shape contextual modulation in the visual cortex*.

## Setup

### Using Docker Compose

Start the development environment:

```bash
docker-compose up -d
docker-compose exec dev bash
```

Run Jupyter notebooks:

```bash
docker-compose exec dev jupyter lab --ip=0.0.0.0 --no-browser
```

### Local Installation

Requirements: PyTorch, PyMC, scikit-image, and the insilico-stimuli package.

See [Dockerfile](Dockerfile) for full dependencies.

## Main Demonstration

The primary notebook is **[experiments/exc_driven_model/binary_custom_mapping.ipynb](experiments/exc_driven_model/binary_custom_mapping.ipynb)**.

### Notebook Goals

1. Build the normative model
2. Generate stimuli (maximum excitatory image, MEI + completing surround, MEI + disruptive surround)
3. Perform posterior inference via sampling to obtain neural activity predictions

## Visualization

See **plots.ipynb** to visualize results and reproduce publication figures.

## Project Structure

- `experiments/` - Main experimental code and notebooks
- `probcs/` - Core package modules
- `data/` - Datasets and stimuli

## Citation

If you use this code, please cite:

Fu et al. 2026. Statistics of natural scenes shape contextual modulation in the visual cortex.
