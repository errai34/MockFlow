# MockFlow: Normalizing Flows for Star-Particle Data

This project implements normalizing flows to learn and generate synthetic star-particle phase-space data (3D position + 3D velocity) from astronomical simulations.

## Features

- **Data Loading**: Load star-particle data from HDF5 files
- **Flow Training**: Train masked autoregressive flows on 6D phase-space data
- **Sample Generation**: Generate new synthetic particles that follow the learned distribution
- **Comprehensive Analysis**: Statistical comparisons, correlation analysis, and visualizations
- **Two Notebooks**: 
  - `src/flow_demo.ipynb`: Demo with synthetic data
  - `train_flow_real_data.ipynb`: Training on real data (5000 particles)

## Quick Start

### Prerequisites

Make sure you have [uv](https://docs.astral.sh/uv/) installed. If not, install it:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup Environment

1. **Clone or navigate to the project directory:**
   ```bash
   cd MockFlow
   ```

2. **Create and activate a virtual environment with uv:**
   ```bash
   # Create virtual environment
   uv venv

   # Activate it (macOS/Linux)
   source .venv/bin/activate

   # Activate it (Windows)
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

### Running the Code

#### Option 1: Jupyter Notebook Interface

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open and run notebooks:**
   - `src/flow_demo.ipynb` - Demo with synthetic data
   - `train_flow_real_data.ipynb` - Training on real data

#### Option 2: Direct Python Usage

```python
from src.flow import (
    read_h5_to_dict,
    prepare_star_particle_dataset,
    train_flow,
    sample,
)
import jax.random as jr

# Load data
star_particles = read_h5_to_dict("data/eden_scaled_Halo088_sunrot0_0kpc200kpcoriginal_particles.h5")

# Prepare dataset (5000 particles)
x_std, median, std = prepare_star_particle_dataset(star_particles, sample_size=5000)

# Train flow
key = jr.key(42)
flow, losses = train_flow(key, x_std)

# Generate new samples
new_samples = sample(flow, key, n_samples=5000, median=median, std=std)
```

## Project Structure

```
MockFlow/
├── data/
│   └── eden_scaled_Halo088_sunrot0_0kpc200kpcoriginal_particles.h5
├── src/
│   ├── flow.py              # Core flow utilities
│   └── flow_demo.ipynb      # Demo notebook with synthetic data
├── train_flow_real_data.ipynb  # Main training notebook
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Key Components

### `src/flow.py`
Core utilities for:
- Loading HDF5 data files
- Preprocessing and sampling particle data
- Building and training normalizing flows
- Generating new samples
- Visualization helpers

### Notebooks
- **`src/flow_demo.ipynb`**: Demonstrates the complete workflow with synthetic data
- **`train_flow_real_data.ipynb`**: Trains on actual star-particle data with comprehensive analysis

## Data Format

The code expects HDF5 files with:
- `pos3`: 3D positions (N × 3 array)
- `vel3`: 3D velocities (N × 3 array)

Where N is the number of particles.

## Output

The training notebook generates:
- **Trained flow model**: Saved as pickle file
- **Generated samples**: Saved as HDF5 file
- **Visualizations**: Marginal distributions, 3D scatter plots, correlation matrices
- **Statistical analysis**: Comprehensive comparison between original and generated data

## Dependencies

- **Core**: numpy, matplotlib, h5py
- **ML**: jax, jaxlib, flowjax
- **Notebooks**: jupyter, ipykernel
- **Optional**: seaborn, scipy, pandas

## Usage Examples

### Training a Flow

```python
# Load and prepare data
star_particles = read_h5_to_dict("data/your_data.h5")
x_std, median, std = prepare_star_particle_dataset(star_particles, sample_size=5000)

# Train
key = jr.key(42)
flow, losses = train_flow(key, x_std, learning_rate=1e-3)
```

### Generating Samples

```python
# Generate new particles
key, subkey = jr.split(key)
new_samples = sample(flow, subkey, n_samples=1000, median=median, std=std)

# Split into position and velocity
positions = new_samples[:, :3]  # First 3 columns
velocities = new_samples[:, 3:] # Last 3 columns
```

### Visualization

```python
from src.flow import plot_marginals

# Compare original vs generated
plot_marginals(
    original_data,
    generated_data,
    dims=(0, 1, 2, 3, 4, 5),
    labels=['X', 'Y', 'Z', 'Vx', 'Vy', 'Vz']
)
```

## Troubleshooting

### Common Issues

1. **JAX Installation Issues**:
   ```bash
   # For CPU-only JAX
   uv pip install jax[cpu]
   
   # For GPU support (CUDA)
   uv pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

2. **HDF5 File Not Found**:
   - Ensure your data file is in the `data/` directory
   - Update the path in the notebook if your data is elsewhere

3. **Memory Issues**:
   - Reduce `sample_size` in `prepare_star_particle_dataset()`
   - Use smaller batch sizes during training

4. **Jupyter Kernel Issues**:
   ```bash
   uv pip install ipykernel
   python -m ipykernel install --user --name=mockflow
   ```

### Environment Verification

Test your setup:

```python
import jax
import flowjax
import h5py
import numpy as np
import matplotlib.pyplot as plt

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("All imports successful!")
```

## Performance Notes

- Training typically takes 2-5 minutes on modern hardware
- Memory usage scales with sample size (~1GB for 5000 particles)
- GPU acceleration available through JAX (optional)

## Contributing

Feel free to submit issues or pull requests for improvements!

## License

This project is for research purposes. Please cite appropriately if used in publications.