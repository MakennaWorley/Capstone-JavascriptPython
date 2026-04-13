"""
System optimization script for high-performance training.
Run this before training to optimize your system for the 3070ti + 64GB RAM setup.
"""

import os
import subprocess
import sys


def set_environment_variables():
	"""Set optimal environment variables for your hardware."""
	print('🔧 Configuring environment variables for high-performance training...')

	# JAX/XLA optimizations for RTX 3070ti
	env_vars = {
		# GPU Memory Management
		'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.85',  # Use 85% of 8GB VRAM
		'XLA_PYTHON_CLIENT_PREALLOCATE': 'true',
		'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
		# Performance optimizations
		'JAX_ENABLE_X64': 'false',  # Use float32 for speed
		'XLA_FLAGS': '--xla_gpu_enable_fast_math=true --xla_gpu_cuda_data_dir=/usr/local/cuda',
		# Multi-threading for your CPU
		'OMP_NUM_THREADS': '20',  # Adjust based on your CPU cores
		'MKL_NUM_THREADS': '20',
		'NUMBA_NUM_THREADS': '20',
		# PyMC optimizations
		'THEANO_FLAGS': 'device=cuda,floatX=float32,force_device=True',
		# Memory optimizations for 64GB RAM
		'PYTHONHASHSEED': '0',  # Reproducibility
	}

	for key, value in env_vars.items():
		os.environ[key] = value
		print(f'  {key} = {value}')

	print('Environment variables configured!')


def check_system_specs():
	"""Check and display system specifications."""
	print('System Specifications:')

	try:
		import psutil

		print(f'  CPU: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count(logical=True)} logical cores')
		print(f'  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB')
		print(f'  Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')
	except ImportError:
		print('  Install psutil for detailed system info: pip install psutil')

	try:
		import jax

		gpu_devices = jax.devices('gpu')
		if gpu_devices:
			print(f'  GPU: {len(gpu_devices)} GPU(s) detected')
			for i, device in enumerate(gpu_devices):
				print(f'    GPU {i}: {device}')
		else:
			print('  GPU: No CUDA GPUs found')
	except ImportError:
		print('  JAX not installed - install for GPU support')


def install_missing_packages():
	"""Install packages needed for optimal performance."""
	packages = ['psutil', 'jax[cuda11_pip]', 'jaxlib']  # Adjust CUDA version as needed

	print('Checking required packages...')
	for package in ['psutil']:  # Only install psutil automatically
		try:
			__import__(package)
			print(f'  {package} is installed')
		except ImportError:
			print(f'  Installing {package}...')
			subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


def optimize_system():
	"""Main optimization function."""
	print('Optimizing system for high-performance Bayesian training')
	print('=' * 60)

	install_missing_packages()
	print()

	check_system_specs()
	print()

	set_environment_variables()
	print()

	print('Recommendations for RTX 3070ti + 64GB RAM:')
	print('  • Use 6-8 chains for optimal GPU utilization')
	print('  • Set cores=16-20 to use most of your CPU')
	print('  • Increase draws/tune to 1500+ for better accuracy')
	print("  • Use 'aggressive' GPU strategy")
	print('  • Monitor GPU memory usage (should use ~6-7GB)')
	print()
	print('System optimization complete! Run your training now.')


if __name__ == '__main__':
	optimize_system()
