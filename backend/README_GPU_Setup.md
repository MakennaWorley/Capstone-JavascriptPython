# GPU Setup Guide for Your PyMC Models

Your code has been updated to support GPU acceleration, but it requires some setup to work properly.

## Quick Test

First, run the GPU setup script to check your current status:

```bash
cd backend
python gpu_setup.py
```

## Step-by-Step GPU Setup

### 1. WSL + NVIDIA Setup

Since you're using WSL with a GPU:

**On Windows:**
1. Install the latest NVIDIA drivers for your GPU from NVIDIA's website
2. Make sure you have WSL 2 (not WSL 1): `wsl --list -v`
3. Update WSL: `wsl --update`

**In WSL:**
```bash
# Check if GPU is visible
nvidia-smi

# Should show your GPU information
```

### 2. Install JAX with CUDA Support

```bash
# Install JAX with CUDA 12 (adjust version if needed)
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Or for CUDA 11:
# pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 3. Test Your Setup

```bash
python gpu_setup.py
```

This will:
- Check NVIDIA drivers
- Test JAX GPU functionality
- Test PyMC with GPU acceleration
- Give you a summary of what's working

## Code Changes Made

Your `BayesianCategoricalDosageClassifier` now:

1. **Automatically detects GPU** - Checks for JAX GPU support on initialization
2. **Optimizes for GPU** - Uses fewer chains and different settings when GPU is available
3. **Falls back gracefully** - Still works on CPU if GPU isn't available
4. **Provides feedback** - Tells you whether it's using GPU or CPU

## Usage Examples

```python
# GPU-enabled (default)
model = BayesianCategoricalDosageClassifier()

# Force CPU usage
model = BayesianCategoricalDosageClassifier(use_gpu=False)

# Custom GPU settings
model = BayesianCategoricalDosageClassifier(
    draws=2000, 
    tune=1000, 
    chains=2,  # GPU works better with fewer chains
    use_gpu=True
)
```

## Expected Performance Improvements

With GPU acceleration, you should see:
- **2-5x faster training** for your Bayesian models
- **Better scaling** with larger datasets
- **More efficient sampling** for complex hierarchical models

## Troubleshooting

### Common Issues:

1. **"No GPU devices found"**
   - Check `nvidia-smi` works in WSL
   - Reinstall NVIDIA drivers on Windows
   - Update WSL: `wsl --update`

2. **"JAX not using GPU"**
   - Try different CUDA versions in JAX installation
   - Check CUDA compatibility with your GPU

3. **"Out of memory"**
   - Reduce batch size or number of chains
   - Your model automatically uses fewer chains on GPU to help with this

4. **Slower than CPU**
   - GPU has overhead; only helps with larger models
   - Try increasing `draws` and `tune` parameters

### Still Not Working?

If GPU setup fails, your models will automatically fall back to CPU mode and work exactly as before. The GPU setup is purely optional for performance improvement.

## Performance Monitoring

Add this to monitor GPU usage during training:

```bash
# In another terminal while training
watch nvidia-smi
```

You should see GPU memory and utilization increase during model fitting.