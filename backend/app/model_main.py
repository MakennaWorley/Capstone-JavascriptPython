from __future__ import annotations

import concurrent.futures
import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

matplotlib.use('Agg')
# Imports from this Repo
import data_preparation
import model_graph_functions
from model_bayesian import BayesianCategoricalDosageClassifier
from model_functions import _flatten_examples, _model_paths, _NoRefitProxy
from model_multi_log_regression import SklearnMultinomialClassifier

# -----------------------------
# Utilities
# -----------------------------


ModelType = Union[BayesianCategoricalDosageClassifier, SklearnMultinomialClassifier]


def _select_model(model_label: str) -> Tuple[Type[ModelType], str]:
	"""Returns the Model Class and a human-readable tag."""
	match model_label:
		case 'bayes_softmax3':
			return (BayesianCategoricalDosageClassifier, 'bayes_softmax3')
		case 'multi_log_regression':
			return (SklearnMultinomialClassifier, 'multi_log_regression')
		case _:
			raise ValueError(f'Unknown model label: {model_label}')


def _run_fold_parallel(args):
	"""Helper function to run a single fold in a separate process."""
	fold_idx, train_idx, val_idx, X_all, y_all, groups_all, model_label = args

	# 1. Slice the data for this specific fold
	X_t, X_v = X_all[train_idx], X_all[val_idx]
	y_t, y_v = y_all[train_idx], y_all[val_idx]
	g_t = groups_all[train_idx]

	# 2. Use your selection helper to get the correct Class
	ModelCls, _ = _select_model(model_label)

	# 3. Dynamic Initialization
	if model_label == 'bayes_softmax3':
		# Bayesian needs specific MCMC parameters for parallel stability
		fold_model = ModelCls(chains=2, draws=500, tune=500, cores=1)
	else:
		# Sklearn model uses standard defaults
		fold_model = ModelCls()

	# 4. Fit and Evaluate
	# Note: groups=g_t is passed to both; Sklearn ignores it safely
	fold_model.fit(X_t, y_t, groups=g_t)

	y_pred = fold_model.predict(X_v)
	mse = np.mean((y_v - y_pred) ** 2)

	return fold_idx, mse


# -----------------------------
# Pipeline
# -----------------------------


def train_with_cross_val(base_name, model_label, prep_cfg, n_splits=5):
	"""
	Performs K-Fold CV using the entirety of a single file.
	"""
	# Load the whole file as one block
	X_all, y_all, groups_all = load_whole_dataset(base_name, prep_cfg)
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

	fold_args = []
	for i, (t_idx, v_idx) in enumerate(kf.split(X_all)):
		fold_args.append((i, t_idx, v_idx, X_all, y_all, groups_all, model_label))

	print(f'\n--- Parallel CV on {base_name} for {model_label} ---')

	# Use built-in multiprocessing Pool
	with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
		results = list(executor.map(_run_fold_parallel, fold_args))

	for fold_idx, mse in sorted(results):
		print(f'Fold {fold_idx + 1} MSE: {mse:.4f}')

	# Return model fit on the whole file
	ModelCls, _ = _select_model(model_label)
	final_model = ModelCls()
	final_model.fit(X_all, y_all, groups=groups_all)
	return final_model


def load_whole_dataset(base_name: str, prep_cfg: data_preparation.PrepConfig):
	"""
	Loads a dataset file and flattens it completely without looking for
	train/val/test sub-splits.
	"""
	# Create a NEW config instance because the original is frozen
	# This copies all settings from prep_cfg but updates the dataset_name
	current_cfg = dataclasses.replace(prep_cfg, dataset_name=base_name)

	# Pass the updated config to prepare_data
	data = data_preparation.prepare_data(current_cfg)

	X_raw = data['X']
	y_raw = data['y']

	X, y = _flatten_examples(X_raw, y_raw)

	if 'groups' in data:
		n_sites = X_raw.shape[1]
		groups = np.repeat(data['groups'].flatten().astype(int), n_sites)
	else:
		groups = np.zeros(X.shape[0], dtype=int)

	return X, y, groups


# this needs to be updated to work with _select_model
def run_custom_pipeline(train_file, val_file, test_file):
	prep_cfg = data_preparation.PrepConfig()

	# 1. TRAINING: Use all data in the training file
	X_train, y_train, g_train = load_whole_dataset(train_file, prep_cfg)
	model = BayesianCategoricalDosageClassifier(cores=1)

	print(f'Phase 1: Training on {train_file}')
	model.fit(X_train, y_train, groups=g_train)

	# 2. VALIDATION: Use cross-validation on the validation file to update the model
	print(f'Phase 2: Updating model with CV on {val_file}')
	# Pass the existing model or refit using the logic in train_with_cross_val
	X_val, y_val, g_val = load_whole_dataset(val_file, prep_cfg)
	model.fit(X_val, y_val, groups=g_val)

	# 3. TESTING: Purely unseen data
	X_test, y_test, _ = load_whole_dataset(test_file, prep_cfg)
	print(f'Phase 3: Testing on {test_file}')
	predictions = model.predict(X_test)

	# Calculate final metrics...
	return predictions


# this needs to be updated to work with _select_model
def train_eval_one(
	train_base: str,
	val_base: str,
	test_base: str,
	model_label: str,
	*,
	prep_cfg: Optional[data_preparation.PrepConfig] = None,
	models_dir: str | Path = 'models',
	force_retrain: bool = False,
	draws: int = 1000,
	tune: int = 1000,
	chains: int = 4,
	target_accept: float = 0.9,
	seed: int = 123,
	cores: int = 8,
) -> Dict[str, Any]:
	if prep_cfg is None:
		prep_cfg = data_preparation.PrepConfig(dataset_name='unused')

	# 1. Setup Model Types
	ModelCls, model_tag = _select_model(model_label)
	paths = _model_paths(models_dir, train_base, model_tag)

	# 2. PHASE 1: INITIAL TRAINING (Using the entire training file)
	X_train, y_train, groups_train = load_whole_dataset(train_base, prep_cfg)

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
		X_resampled, y_resampled, groups_resampled = data_preparation.resample_training_data(X_train, y_train, groups_train)

		# Handle different constructor signatures
		if model_label == 'bayes_softmax3':
			model = ModelCls(draws=draws, tune=tune, chains=chains, target_accept=target_accept, random_seed=seed, cores=cores)
		else:
			model = ModelCls(random_seed=seed)

		model.fit(X_resampled, y_resampled, groups=groups_resampled)
		trained = True

	# 3. PHASE 2: CROSS-VALIDATION UPDATE (Using the entire validation file)
	print(f'--- Phase 2: Updating {model_tag} with CV on {val_base} ---')
	# This calls your KFold logic on the dedicated validation file
	model = train_with_cross_val(val_base, model_label, prep_cfg)

	# Save immediately after updating
	model.save(paths, extra_meta={'train_src': train_base, 'val_src': val_base})

	# 4. PHASE 3: TESTING (Evaluating on unseen data)
	print(f'--- Phase 3: Final Testing {model_tag} on {test_base} ---')
	X_test, y_test, groups_test = load_whole_dataset(test_base, prep_cfg)

	test_metrics = model_graph_functions.evaluate_and_graph_clf(model, X_test, y_test, groups=groups_test, name=f'{model_tag}', graph=True)

	if paths['graph_test']:
		plt.savefig(paths['graph_test'])
		plt.close()

	# Confusion Matrix
	y_pred_cm = model.predict_class(X_test, groups=groups_test)
	model_graph_functions.plot_confusion_matrix(y_true=y_test, y_pred=y_pred_cm, name=f'{model_tag} Confusion Matrix', save_path=paths['graph_cm'])

	return {'trained': trained, 'test_metrics': test_metrics, 'paths': {k: str(v) for k, v in paths.items() if k != 'dir'}}


def test_on_new_data(model, dataset_name: str, prep_cfg: Optional[data_preparation.PrepConfig] = None):
	"""
	Loads a new dataset, prepares it using the model's original scaling params,
	and returns predictions and metrics without any training.
	"""
	if prep_cfg is None:
		prep_cfg = data_preparation.PrepConfig(dataset_name=dataset_name)

	# 1. Load the new data
	new_data = data_preparation.prepare_data(dataset_name, prep_cfg)
	X_raw, y_raw = _flatten_examples(new_data['test']['X'], new_data['test']['y'])

	# 2. Wrap the model in the NoRefitProxy to ensure .fit() cannot be called
	proxy = _NoRefitProxy(model)

	print(f'\n=== Testing on New Dataset: {dataset_name} ===')

	# 3. Use the existing evaluation utility to get metrics and plots
	# This uses model.predict() internally, which applies stored feature_mean_ and feature_std_
	if hasattr(model, 'predict_class'):
		metrics = model_graph_functions.evaluate_and_graph_clf(proxy, X_raw, y_raw, name=f'External_Test_{dataset_name}', graph=True)
	else:
		metrics = model_graph_functions.evaluate_and_graph_reg(proxy, X_raw, y_raw, name=f'External_Test_{dataset_name}', graph=True)
	return metrics


def train_eval_comparison(train_f, val_f, test_f):
	"""Runs both models for the capstone comparison."""
	results = {}
	for label in ['bayes_softmax3', 'multi_log_regression']:
		results[label] = train_eval_one(train_f, val_f, test_f, label)
	return results


if __name__ == '__main__':
	train_eval_comparison('testing.training', 'testing.validation', 'testing.testing')
