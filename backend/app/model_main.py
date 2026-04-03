from __future__ import annotations

import csv
import dataclasses
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

if __name__ == '__main__' and __package__ is None:
	current_dir = Path(__file__).parent
	backend_dir = current_dir.parent
	capstone_dir = backend_dir.parent
	sys.path.insert(0, str(capstone_dir))
	__package__ = 'backend.app'

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import KFold

matplotlib.use('Agg')

# Imports from this Repo
from .data_preparation import PrepConfig, prepare_data, resample_training_data
from .model_bayesian import BayesianCategoricalDosageClassifier
from .model_dnn import DNNDosageClassifier
from .model_functions import flatten_examples, model_paths
from .model_gnn import GNNDosageClassifier
from .model_graph_functions import evaluate_and_graph_clf, plot_confusion_matrix
from .model_hmm import HMMDosageClassifier
from .model_multi_log_regression import SklearnMultinomialClassifier

# Load environment variables
load_dotenv()
DATASETS_DIR = os.getenv('DATASETS_DIR')
PROTECTED_DATASETS_DIR = os.getenv('PROTECTED_DATASETS_DIR')
MODELS_DIR = os.getenv('MODELS_DIR')
IMAGES_DIR = os.getenv('IMAGES_DIR')


def check_gpu_status():
	"""Check and report GPU availability for models."""
	try:
		import jax

		gpu_devices = jax.devices('gpu')
		if gpu_devices:
			print(f'🚀 GPU acceleration available: {len(gpu_devices)} GPU(s) detected')
			for i, device in enumerate(gpu_devices):
				print(f'  GPU {i}: {device}')
		else:
			print('💻 Running on CPU (no GPU devices found)')

		# Update this for your computer, this was running on my i9-12900k
		os.environ['OMP_NUM_THREADS'] = '12'
		os.environ['MKL_NUM_THREADS'] = '12'
	except ImportError:
		print('💻 Running on CPU (JAX not installed)')
	except Exception as e:
		print(f'💻 Running on CPU (GPU check failed: {e})')


def get_optimal_training_config():
	"""Auto-detect system specs and return optimal training configuration."""
	import os

	try:
		import psutil
	except ImportError:
		print('Warning: psutil not installed. Using conservative defaults.')
		psutil = None

	if psutil:
		# Detect CPU cores
		physical_cores = psutil.cpu_count(logical=False)
		logical_cores = psutil.cpu_count(logical=True)

		# Detect RAM
		total_ram_gb = psutil.virtual_memory().total / (1024**3)
	else:
		# Fallback to os module
		logical_cores = os.cpu_count() or 4
		physical_cores = logical_cores // 2  # Estimate
		total_ram_gb = 16  # Conservative estimate

	# Detect GPU
	has_gpu = False
	try:
		import jax

		has_gpu = len(jax.devices('gpu')) > 0
	except:
		pass

	print('\n=== System Configuration ===')
	print(f'CPU: {physical_cores} physical cores, {logical_cores} logical cores')
	print(f'RAM: {total_ram_gb:.1f} GB')
	print(f'GPU: {"Available" if has_gpu else "Not available"}')

	if has_gpu and total_ram_gb >= 32:  # High-end system
		optimal_chains = min(8, physical_cores)  # More chains for GPU
		optimal_cores = min(logical_cores - 2, 20)  # Use most cores but leave some for system
		optimal_draws = 1500  # More samples for better accuracy
		optimal_tune = 1500
		strategy = 'aggressive'
		print('🚀 High-end system detected: Using aggressive optimization')
	elif has_gpu:  # GPU but less RAM
		optimal_chains = min(6, physical_cores)
		optimal_cores = min(logical_cores - 1, 12)
		optimal_draws = 1200
		optimal_tune = 1200
		strategy = 'aggressive'
		print('🚀 GPU system detected: Using moderate optimization')
	else:  # CPU only
		optimal_chains = min(4, physical_cores)
		optimal_cores = min(logical_cores, 8)
		optimal_draws = 1000
		optimal_tune = 1000
		strategy = 'safe'
		print('💻 CPU-only system: Using conservative settings')

	config = {'chains': optimal_chains, 'cores': optimal_cores, 'draws': optimal_draws, 'tune': optimal_tune, 'gpu_strategy': strategy}

	print(f'Optimal config: {config}')
	return config


# -----------------------------
# Utilities
# -----------------------------


ModelType = Union[BayesianCategoricalDosageClassifier, SklearnMultinomialClassifier, HMMDosageClassifier, GNNDosageClassifier]


class OutputLogger:
	"""
	Context manager to capture stdout and save it to a file while also printing to terminal.
	"""

	def __init__(self, log_file_path: str | Path):
		self.log_file_path = Path(log_file_path)
		self.terminal = sys.stdout
		self.log_buffer = StringIO()

	def write(self, message):
		# Write to both terminal and buffer
		self.terminal.write(message)
		self.log_buffer.write(message)

	def flush(self):
		# Flush both outputs
		self.terminal.flush()
		self.log_buffer.flush()

	def __enter__(self):
		sys.stdout = self
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		# Restore original stdout
		sys.stdout = self.terminal

		# Write buffer contents to file
		self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
		with open(self.log_file_path, 'w', encoding='utf-8') as f:
			f.write(self.log_buffer.getvalue())

		self.log_buffer.close()


def _select_model(model_label: str) -> Tuple[Type[ModelType], str]:
	"""
	Returns the Model Class and a human-readable tag.
	"""
	match model_label:
		case 'bayes_softmax3':
			return (BayesianCategoricalDosageClassifier, 'bayes_softmax3')
		case 'multi_log_regression':
			return (SklearnMultinomialClassifier, 'multi_log_regression')
		case 'hmm_dosage':
			return (HMMDosageClassifier, 'hmm_dosage')
		case 'dnn_dosage':
			return (DNNDosageClassifier, 'dnn_dosage')
		case 'gnn_dosage':
			return (GNNDosageClassifier, 'gnn_dosage')
		case _:
			raise ValueError(f'Unknown model label: {model_label}')


def _run_fold_parallel(args):
	"""
	Helper function to run a single fold in a separate process.
	"""
	fold_idx, X_t, X_v, y_t, y_v, g_t, model_label = args

	ModelCls, _ = _select_model(model_label)

	if model_label == 'bayes_softmax3':
		# Bayesian with optimized settings for CV (faster but still accurate)
		fold_model = ModelCls(chains=2, draws=500, tune=500, cores=2)
	elif model_label == 'hmm_dosage':
		# HMM with optimized settings for CV
		fold_model = ModelCls(n_iter=20, verbose=False)
	elif model_label == 'dnn_dosage':
		# DNN with optimized settings for CV
		fold_model = ModelCls(epochs=50, verbose=False, early_stopping_patience=5)
	elif model_label == 'gnn_dosage':
		# GNN with optimized settings for CV
		fold_model = ModelCls(epochs=50, verbose=False, early_stopping_patience=5)
	else:
		fold_model = ModelCls()

	# X_t and y_t are already resampled by the caller
	fold_model.fit(X_t, y_t, groups=g_t)

	y_pred = fold_model.predict(X_v)
	mse = np.mean((y_v - y_pred) ** 2)

	return fold_idx, mse


# -----------------------------
# Pipeline
# -----------------------------


def update_models_csv(model_name: str, model_type: str, csv_path: str | Path = 'models.csv') -> None:
	"""
	Updates the models.csv file with a new model entry.
	If the combination already exists, it won't add a duplicate.
	"""
	csv_file = Path(csv_path)
	existing_entries = set()

	# Read existing entries if file exists
	if csv_file.exists():
		with open(csv_file, 'r', newline='', encoding='utf-8') as f:
			reader = csv.DictReader(f)
			for row in reader:
				existing_entries.add((row['model_name'], row['model_type']))

	# Check if entry already exists
	if (model_name, model_type) in existing_entries:
		print(f'Model entry ({model_name}, {model_type}) already exists in {csv_path}')
		return

	# Add new entry
	file_exists = csv_file.exists()
	with open(csv_file, 'a', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=['model_name', 'model_type'])

		# Write header if file is new
		if not file_exists:
			writer.writeheader()

		writer.writerow({'model_name': model_name, 'model_type': model_type})
		print(f'Added model entry to {csv_path}: ({model_name}, {model_type})')


def evaluate_with_cross_val(val_base_name, model_label, prep_cfg, existing_model, n_splits=5):
	"""
	Evaluates an existing model using K-Fold CV on validation data.
	Returns the validation data (X, y, groups) and average MSE.
	"""
	# Load the validation file
	X_val, y_val, groups_val = load_whole_dataset(val_base_name, prep_cfg)
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

	print(f'\n--- Cross-Validation Evaluation on {val_base_name} ---')
	# Evaluate existing model performance using CV
	fold_mses = []
	for i, (_, v_idx) in enumerate(kf.split(X_val)):
		# Test existing model on each fold
		y_pred_fold = existing_model.predict(X_val[v_idx], groups=groups_val[v_idx])
		mse_fold = np.mean((y_val[v_idx] - y_pred_fold) ** 2)
		fold_mses.append(mse_fold)
		print(f'  Fold {i + 1} Validation MSE: {mse_fold:.4f}')

	avg_mse = np.mean(fold_mses)
	print(f'Average Validation MSE: {avg_mse:.4f}')

	return X_val, y_val, groups_val, avg_mse


def load_whole_dataset(base_name: str, prep_cfg: PrepConfig):
	"""
	Loads a dataset file and flattens it completely without looking for
	train/val/test sub-splits.
	"""
	# Copies all settings from prep_cfg but updates the dataset_name
	current_cfg = dataclasses.replace(prep_cfg, dataset_name=base_name)

	# Pass the updated config to prepare_data
	data = prepare_data(current_cfg)

	X_raw = data['X']
	y_raw = data['y']

	X, y = flatten_examples(X_raw, y_raw)

	if 'groups' in data:
		n_sites = X_raw.shape[1]
		groups = np.repeat(data['groups'].flatten().astype(int), n_sites)
	else:
		groups = np.zeros(X.shape[0], dtype=int)

	return X, y, groups


def train_eval(
	train_base: str,
	val_base: str,
	test_base: str,
	model_label: str,
	*,
	prep_cfg: Optional[PrepConfig] = None,
	models_dir: str | Path = MODELS_DIR,
	images_dir: str | Path = IMAGES_DIR,
	datasets_dir: str | Path = PROTECTED_DATASETS_DIR,
	force_retrain: bool = False,
	draws: int = 1000,
	tune: int = 1000,
	chains: int = 4,
	target_accept: float = 0.99,
	seed: Optional[int] = None,
	cores: int = 8,
	optimize_system_resources: bool = True,
) -> Dict[str, Any]:
	if seed is None:
		seed = int(np.random.SeedSequence().entropy % (2**32))

	if prep_cfg is None:
		prep_cfg = PrepConfig(dataset_name='unused', datasets_dir=str(datasets_dir))

	# Optimize system resources if requested (and not already done)
	if optimize_system_resources and model_label == 'bayes_softmax3':
		try:
			from optimize_system import set_environment_variables

			set_environment_variables()
		except ImportError:
			pass  # Continue without optimization

	# 1. Setup Model Types
	ModelCls, model_tag = _select_model(model_label)
	paths = model_paths(models_dir, train_base, model_tag)

	# Create images directory
	images_path = Path(images_dir)
	images_path.mkdir(parents=True, exist_ok=True)

	# Setup models.csv path in the models directory
	models_csv_path = Path(models_dir) / 'models.csv'

	# Logic: Bayesian needs meta AND idata; Sklearn only needs meta
	exists_check = paths['meta'].exists()
	if model_label == 'bayes_softmax3':
		exists_check = exists_check and paths['idata'].exists()

	if (not force_retrain) and exists_check:
		print(f'Loading existing {model_tag} from {paths["meta"]}')
		model = ModelCls.load(paths)
		trained = False
	else:
		print(f'--- Phase 1: Training {model_tag} on {train_base} ---')
		# 2. PHASE 1: INITIAL TRAINING (Using the entire training file)
		X_train, y_train, groups_train = load_whole_dataset(train_base, prep_cfg)

		# Apply your resampling logic here for training
		X_resampled, y_resampled, groups_resampled = resample_training_data(X_train, y_train, groups_train)

		# Handle different constructor signatures with auto-optimized settings
		if model_label == 'bayes_softmax3':
			optimal_config = get_optimal_training_config()
			model = ModelCls(
				draws=optimal_config['draws'],
				tune=optimal_config['tune'],
				chains=optimal_config['chains'],
				target_accept=target_accept,
				random_seed=seed,
				cores=optimal_config['cores'],
				gpu_strategy=optimal_config['gpu_strategy'],
			)
		elif model_label == 'hmm_dosage':
			model = ModelCls(n_iter=20, random_seed=seed, use_gpu=True, verbose=True)
		elif model_label == 'dnn_dosage':
			model = ModelCls(hidden_dims=(256, 128, 64), epochs=100, random_seed=seed, use_gpu=True, verbose=True, early_stopping_patience=10)
		elif model_label == 'gnn_dosage':
			model = ModelCls(hidden_dims=(256, 128, 64), epochs=100, random_seed=seed, use_gpu=True, verbose=True, early_stopping_patience=10)
		else:
			model = ModelCls(random_seed=seed)

		model.fit(X_resampled, y_resampled, groups=groups_resampled)
		trained = True

	# 3. PHASE 2: CROSS-VALIDATION EVALUATION + RETRAINING (Using training + validation)
	print(f'--- Phase 2: Evaluating on {val_base} and retraining with combined data ---')

	# Evaluate the model on validation data using cross-validation
	X_val, y_val, groups_val, avg_val_mse = evaluate_with_cross_val(val_base, model_label, prep_cfg, existing_model=model)

	# Now combine training + validation data and retrain the model
	print('  Combining training and validation data for final model training...')
	X_combined = np.vstack([X_resampled, X_val])
	y_combined = np.concatenate([y_resampled, y_val])
	groups_combined = np.concatenate([groups_resampled, groups_val])

	# Resample the combined data
	X_combined_resampled, y_combined_resampled, groups_combined_resampled = resample_training_data(X_combined, y_combined, groups_combined)

	print(f'  Retraining {model_tag} on combined training + validation data...')
	# Reinitialize and train on combined data with auto-optimized settings
	if model_label == 'bayes_softmax3':
		optimal_config = get_optimal_training_config()
		model = ModelCls(
			draws=optimal_config['draws'],
			tune=optimal_config['tune'],
			chains=optimal_config['chains'],
			target_accept=target_accept,
			random_seed=seed,
			cores=optimal_config['cores'],
			gpu_strategy=optimal_config['gpu_strategy'],
		)
	elif model_label == 'hmm_dosage':
		model = ModelCls(n_iter=20, random_seed=seed, use_gpu=True, verbose=True)
	elif model_label == 'dnn_dosage':
		model = ModelCls(hidden_dims=(256, 128, 64), epochs=100, random_seed=seed, use_gpu=True, verbose=True, early_stopping_patience=10)
	elif model_label == 'gnn_dosage':
		model = ModelCls(hidden_dims=(256, 128, 64), epochs=100, random_seed=seed, use_gpu=True, verbose=True, early_stopping_patience=10)
	else:
		model = ModelCls(random_seed=seed)

	model.fit(X_combined_resampled, y_combined_resampled, groups=groups_combined_resampled)

	# Save immediately after retraining
	model.save(paths, extra_meta={'train_src': train_base, 'val_src': val_base, 'avg_val_mse': float(avg_val_mse)})

	# Update models.csv with the trained model
	update_models_csv(train_base, model_tag, csv_path=models_csv_path)

	# 4. PHASE 3: TESTING (Evaluating on unseen data)
	# Setup log file for test output
	log_file_path = paths['dir'] / f'{train_base}.{model_tag}.test_log.txt'

	with OutputLogger(log_file_path):
		print(f'--- Phase 3: Final Testing {model_tag} on {test_base} ---')
		X_test, y_test, groups_test = load_whole_dataset(test_base, prep_cfg)

		test_metrics = evaluate_and_graph_clf(model, X_test, y_test, groups=groups_test, name=f'{model_tag}_test_{test_base}', graph=True)

		# Create graph paths based on the test dataset name
		test_graph_path = images_path / f'{model_tag}_test_{test_base}.png'
		test_cm_path = images_path / f'{model_tag}_test_{test_base}_confusion.png'

		if test_graph_path:
			plt.savefig(test_graph_path)
			plt.close()
			print(f'Saved test graph to {test_graph_path}')

		# Confusion Matrix
		y_pred_cm = model.predict_class(X_test, groups=groups_test)
		plot_confusion_matrix(y_true=y_test, y_pred=y_pred_cm, name=f'{model_tag} {test_base}', save_path=test_cm_path)
		print(f'Saved confusion matrix to {test_cm_path}')
		print(f'\nTest log saved to {log_file_path}')

	return {
		'trained': trained,
		'test_metrics': test_metrics,
		'paths': {'meta': str(paths['meta']), 'graph_test': str(test_graph_path), 'graph_cm': str(test_cm_path), 'model_dir': str(paths['dir'])},
	}


def test_on_new_data(
	test_base: str,
	model_type: str,
	model_name: str,
	*,
	prep_cfg: Optional[PrepConfig] = None,
	models_dir: str | Path = MODELS_DIR,
	images_dir: str | Path = IMAGES_DIR,
	datasets_dir: str | Path = DATASETS_DIR,
) -> Dict[str, Any]:
	"""
	Loads a new dataset and applies an existing trained model to it.
	The model must have been previously trained using train_eval() with the specified model_name.
	"""
	if prep_cfg is None:
		prep_cfg = PrepConfig(dataset_name='unused', datasets_dir=str(datasets_dir))

	# 1. Setup Model Types and Paths
	ModelCls, model_tag = _select_model(model_type)
	paths = model_paths(models_dir, model_name, model_tag)

	# Create images directory
	images_path = Path(images_dir)
	images_path.mkdir(parents=True, exist_ok=True)

	# 2. Check if Model Exists
	exists_check = paths['meta'].exists()
	if model_type == 'bayes_softmax3':
		exists_check = exists_check and paths['idata'].exists()

	if not exists_check:
		raise FileNotFoundError(f'Model not found. Expected files at {paths["meta"]}. Please train the model first using train_eval().')

	# 3. Load the Pre-trained Model
	print(f'Loading existing {model_tag} from {paths["meta"]}')
	model = ModelCls.load(paths)

	# Setup log file for test output
	log_file_path = paths['dir'] / f'{model_name}.{model_tag}.test_{test_base}.txt'

	with OutputLogger(log_file_path):
		# 4. Prepare Test Data
		print(f'--- Testing {model_tag} on {test_base} ---')
		X_test, y_test, groups_test = load_whole_dataset(test_base, prep_cfg)

		# 5. Predict and Evaluate
		test_metrics = evaluate_and_graph_clf(model, X_test, y_test, groups=groups_test, name=f'{model_tag}_test_{test_base}', graph=True)

		# 6. Save Graphs
		# Create paths for this specific test evaluation
		test_graph_path = images_path / f'{model_tag}_test_{test_base}_single.png'
		test_cm_path = images_path / f'{model_tag}_test_{test_base}_confusion_single.png'

		if test_graph_path:
			plt.savefig(test_graph_path)
			plt.close()
			print(f'Saved test graph to {test_graph_path}')

		# Confusion Matrix
		y_pred_cm = model.predict_class(X_test, groups=groups_test)
		plot_confusion_matrix(y_true=y_test, y_pred=y_pred_cm, name=f'{model_tag} {test_base}', save_path=test_cm_path)
		print(f'Saved confusion matrix to {test_cm_path}')
		print(f'\nTest log saved to {log_file_path}')

	return {
		'test_metrics': test_metrics,
		'paths': {'graph_test': str(test_graph_path), 'graph_cm': str(test_cm_path), 'model_dir': str(paths['dir'])},
	}


def train_eval_all(train_f, val_f, test_f, *, datasets_dir: str | Path = PROTECTED_DATASETS_DIR):
	"""Runs both models for the capstone comparison using a specific datasets_dir."""
	print('=== Model Training Comparison ===')

	# Import and run system optimization
	try:
		from optimize_system import optimize_system

		optimize_system()
	except ImportError:
		print('Warning: optimize_system.py not found. Continuing with default settings.')
		check_gpu_status()

	print()

	results = {}
	for label in ['bayes_softmax3', 'multi_log_regression', 'hmm_dosage', 'dnn_dosage', 'gnn_dosage']:
		results[label] = train_eval(train_f, val_f, test_f, label, datasets_dir=datasets_dir)
	return results


if __name__ == '__main__':
	# Force use of the protected datasets directory for training runs
	# train_eval_all('tiny.training', 'tiny.validation', 'tiny.testing', datasets_dir=PROTECTED_DATASETS_DIR)
	# train_eval_all('small.training', 'small.validation', 'small.testing', datasets_dir=PROTECTED_DATASETS_DIR)
	# train_eval_all('medium.training', 'medium.validation', 'medium.testing', datasets_dir=PROTECTED_DATASETS_DIR)

	print(test_on_new_data('public', 'bayes_softmax3', 'tiny.training'))
	# print(test_on_new_data('small.testing', 'multi_log_regression', 'small.training'))
	# print(test_on_new_data('medium.testing', 'bayes_softmax3', 'medium.training'))
	# print(test_on_new_data('medium.testing', 'multi_log_regression', 'medium.training'))
