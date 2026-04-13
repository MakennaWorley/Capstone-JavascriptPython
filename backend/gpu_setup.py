"""
GPU Setup and Testing Script for PyMC Models

This script helps you:
1. Check if GPU support is available
2. Install the necessary GPU dependencies
3. Test GPU acceleration with a simple model
"""

import subprocess
import sys
from typing import Tuple


def check_gpu_drivers() -> Tuple[bool, str]:
	"""Check if NVIDIA GPU drivers are available."""
	try:
		result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
		if result.returncode == 0:
			return True, result.stdout
		else:
			return False, 'nvidia-smi command failed'
	except FileNotFoundError:
		return False, 'nvidia-smi not found - NVIDIA drivers may not be installed'


def check_cuda() -> Tuple[bool, str]:
	"""Check CUDA installation."""
	try:
		result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
		if result.returncode == 0:
			return True, result.stdout
		else:
			return False, 'nvcc command failed'
	except (FileNotFoundError, PermissionError, OSError) as e:
		return False, f'nvcc not found or not accessible - CUDA toolkit may not be installed ({type(e).__name__}: {e})'


def install_jax_gpu() -> bool:
	"""Install JAX with CUDA support."""
	print('Installing JAX with CUDA support...')
	try:
		# Install JAX with CUDA 12 support (adjust version as needed)
		subprocess.check_call(
			[sys.executable, '-m', 'pip', 'install', 'jax[cuda12]', '-f', 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html']
		)
		print('JAX with CUDA support installed successfully!')
		return True
	except subprocess.CalledProcessError as e:
		print(f'Failed to install JAX with CUDA: {e}')
		return False


def test_jax_gpu() -> Tuple[bool, str]:
	"""Test if JAX can use GPU."""
	try:
		import jax
		import jax.numpy as jnp

		# Check available devices
		devices = jax.devices()
		gpu_devices = jax.devices('gpu')

		info = f'JAX devices: {devices}\n'
		info += f'GPU devices: {gpu_devices}\n'

		if gpu_devices:
			# Test a simple computation
			x = jnp.array([1.0, 2.0, 3.0])
			y = jnp.dot(x, x)
			info += f'GPU test computation successful: {y}\n'
			info += f'Result device: {y.device}\n'
			return True, info
		else:
			return False, info + 'No GPU devices found in JAX'

	except ImportError:
		return False, 'JAX not installed'
	except Exception as e:
		return False, f'JAX GPU test failed: {e}'


def test_pymc_gpu() -> Tuple[bool, str]:
	"""Test if PyMC can use GPU acceleration."""
	try:
		import jax
		import numpy as np
		import pymc as pm

		# Simple test model
		np.random.seed(42)
		data = np.random.randn(100)

		with pm.Model() as model:
			mu = pm.Normal('mu', mu=0, sigma=1)
			sigma = pm.HalfNormal('sigma', sigma=1)
			obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=data)

			# Sample with fewer draws for testing
			trace = pm.sample(100, tune=100, chains=2, cores=1, return_inferencedata=True)

		return True, f'PyMC GPU test successful! Used backend: {pm.PLATFORM}'

	except ImportError as e:
		return False, f'Missing dependency: {e}'
	except Exception as e:
		return False, f'PyMC GPU test failed: {e}'


def main():
	"""Main GPU setup and testing function."""
	print('=== GPU Setup and Testing for PyMC ===\n')

	# 1. Check GPU drivers
	print('1. Checking GPU drivers...')
	gpu_ok, gpu_info = check_gpu_drivers()
	if gpu_ok:
		print('[OK] NVIDIA drivers found:')
		print(gpu_info[:200] + '...' if len(gpu_info) > 200 else gpu_info)
	else:
		print('[FAIL] NVIDIA drivers issue:')
		print(gpu_info)
		print('\nTo install NVIDIA drivers in WSL:')
		print('1. Install NVIDIA drivers on Windows')
		print('2. Update WSL: wsl --update')
		print('3. No additional installation needed in WSL')
		return

	# 2. Check CUDA
	print('\n2. Checking CUDA...')
	cuda_ok, cuda_info = check_cuda()
	if cuda_ok:
		print('[OK] CUDA found:')
		print(cuda_info.split('\n')[0])  # Just the version line
	else:
		print('[INFO] CUDA toolkit not found (not required for JAX):')
		print(cuda_info[:100] + '...' if len(cuda_info) > 100 else cuda_info)

	# 3. Test JAX GPU
	print('\n3. Testing JAX GPU support...')
	jax_ok, jax_info = test_jax_gpu()
	if not jax_ok:
		print('[FAIL] JAX GPU not working:')
		print(jax_info)
		print('\nInstalling JAX with GPU support...')
		if install_jax_gpu():
			jax_ok, jax_info = test_jax_gpu()

	if jax_ok:
		print('[OK] JAX GPU working:')
		print(jax_info)
	else:
		print('[FAIL] JAX GPU still not working after installation attempt')
		return

	# 4. Test PyMC GPU
	print('\n4. Testing PyMC with GPU...')
	pymc_ok, pymc_info = test_pymc_gpu()
	if pymc_ok:
		print('[OK] PyMC GPU test passed:')
		print(pymc_info)
	else:
		print('[FAIL] PyMC GPU test failed:')
		print(pymc_info)

	# 5. Summary
	print('\n=== Summary ===')
	print(f'GPU Drivers: {"OK" if gpu_ok else "FAIL"}')
	print(f'CUDA: {"OK" if cuda_ok else "INFO"} (optional)')
	print(f'JAX GPU: {"OK" if jax_ok else "FAIL"}')
	print(f'PyMC GPU: {"OK" if pymc_ok else "FAIL"}')

	if gpu_ok and jax_ok and pymc_ok:
		print('\nGPU acceleration is ready to use!')
		print('\nYour models will automatically use GPU when you:')
		print('1. Create models with use_gpu=True (default)')
		print('2. Run training - PyMC will automatically detect and use GPU')
	else:
		print('\nGPU acceleration is not fully working')
		print('Your models will run on CPU (which still works fine)')


if __name__ == '__main__':
	main()
