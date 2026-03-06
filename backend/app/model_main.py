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
from model_functions import _flatten_examples, _model_paths
from model_multi_log_regression import SklearnMultinomialClassifier

# -----------------------------
# Utilities
# -----------------------------


ModelType = Union[BayesianCategoricalDosageClassifier, SklearnMultinomialClassifier]


def _select_model(model_label: str) -> Tuple[Type[ModelType], str]:
	"""
	Returns the Model Class and a human-readable tag.
	"""
	match model_label:
		case 'bayes_softmax3':
			return (BayesianCategoricalDosageClassifier, 'bayes_softmax3')
		case 'multi_log_regression':
			return (SklearnMultinomialClassifier, 'multi_log_regression')
		case _:
			raise ValueError(f'Unknown model label: {model_label}')


def _run_fold_parallel(args):
	"""
	Helper function to run a single fold in a separate process.
	"""
	fold_idx, X_t, X_v, y_t, y_v, g_t, model_label = args

	ModelCls, _ = _select_model(model_label)

	if model_label == 'bayes_softmax3':
		# Reduced chains/draws for faster CV iterations
		fold_model = ModelCls(chains=2, draws=500, tune=500, cores=1)
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


def train_with_cross_val(base_name, model_label, prep_cfg, n_splits=5, existing_model=None):
	"""
	Performs K-Fold CV using the entirety of a single file.
	If existing_model is provided, updates it with the new data instead of training from scratch.
	"""
	# Load the whole file as one block
	X_all, y_all, groups_all = load_whole_dataset(base_name, prep_cfg)
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

	if existing_model is not None:
		print(f'\n--- Updating existing model with CV on {base_name} ---')
		# Evaluate existing model performance first
		fold_mses = []
		for i, (_, v_idx) in enumerate(kf.split(X_all)):
			# Test existing model on each fold
			y_pred_fold = existing_model.predict(X_all[v_idx], groups=groups_all[v_idx])
			mse_fold = np.mean((y_all[v_idx] - y_pred_fold) ** 2)
			fold_mses.append(mse_fold)
			print(f'  Fold {i + 1} Pre-update MSE: {mse_fold:.4f}')

		avg_mse_before = np.mean(fold_mses)
		print(f'Average Pre-update MSE: {avg_mse_before:.4f}')

		# Update model with all validation data (resampled)
		X_update, y_update, g_update = data_preparation.resample_training_data(X_all, y_all, groups_all)

		# Different update strategies based on model type
		if model_label == 'bayes_softmax3':
			# For Bayesian models, we need to re-fit with new data
			# The existing model's posterior could be used as prior, but for now we'll retrain
			print('  Re-training Bayesian model with validation data...')
			existing_model.fit(X_update, y_update, groups=g_update)
		elif hasattr(existing_model, 'partial_fit'):
			# Incremental update for models that support it (like sklearn online learners)
			print('  Incrementally updating model...')
			existing_model.partial_fit(X_update, y_update, groups=g_update)
		else:
			# Re-fit with new data for other models
			print('  Re-fitting model with validation data...')
			existing_model.fit(X_update, y_update, groups=g_update)

		# Test updated model performance
		updated_mses = []
		for i, (_, v_idx) in enumerate(kf.split(X_all)):
			y_pred_fold = existing_model.predict(X_all[v_idx], groups=groups_all[v_idx])
			mse_fold = np.mean((y_all[v_idx] - y_pred_fold) ** 2)
			updated_mses.append(mse_fold)
			print(f'  Fold {i + 1} Post-update MSE: {mse_fold:.4f}')

		avg_mse_after = np.mean(updated_mses)
		print(f'Average Post-update MSE: {avg_mse_after:.4f}')
		print(f'MSE Change: {avg_mse_after - avg_mse_before:.4f}')

		return existing_model

	else:
		# Original behavior - train new model from scratch
		fold_args = []
		for i, (t_idx, v_idx) in enumerate(kf.split(X_all)):
			# Resample ONLY the training portion of the fold
			X_fold_train, y_fold_train, g_fold_train = data_preparation.resample_training_data(X_all[t_idx], y_all[t_idx], groups_all[t_idx])

			# Validation portion (v_idx) stays raw to give an honest MSE
			fold_args.append((i, X_fold_train, X_all[v_idx], y_fold_train, y_all[v_idx], g_fold_train, model_label))

	print(f'\n--- Parallel CV on {base_name} for {model_label} ---')

	# Use built-in multiprocessing Pool
	with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
		results = list(executor.map(_run_fold_parallel, fold_args))

	for fold_idx, mse in sorted(results):
		print(f'Fold {fold_idx + 1} MSE: {mse:.4f}')

	# Return model fit on the whole file
	X_final, y_final, g_final = data_preparation.resample_training_data(X_all, y_all, groups_all)
	ModelCls, _ = _select_model(model_label)
	final_model = ModelCls()
	final_model.fit(X_final, y_final, groups=g_final)
	return final_model


def load_whole_dataset(base_name: str, prep_cfg: data_preparation.PrepConfig):
	"""
	Loads a dataset file and flattens it completely without looking for
	train/val/test sub-splits.
	"""
	# Copies all settings from prep_cfg but updates the dataset_name
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


def run_custom_pipeline(
	train_file: str,
	val_file: str,
	test_file: str,
	model_label: str,
	*,
	prep_cfg: Optional[data_preparation.PrepConfig] = None,
	models_dir: str | Path = 'models',
	draws: int = 1000,
	tune: int = 1000,
	chains: int = 4,
	target_accept: float = 0.99,
	seed: int = 123,
	cores: int = 1,
) -> Dict[str, Any]:
	"""
	Custom 3-phase pipeline: train, validate, test.
	"""
	if prep_cfg is None:
		prep_cfg = data_preparation.PrepConfig(dataset_name='unused')

	# Setup model type
	ModelCls, model_tag = _select_model(model_label)
	paths = _model_paths(models_dir, train_file, model_tag)

	# 1. TRAINING: Use all data in the training file and balance counts
	print(f'Phase 1: Training {model_tag} on {train_file}')
	X_raw, y_raw, g_raw = load_whole_dataset(train_file, prep_cfg)
	X_train, y_train, g_train = data_preparation.resample_training_data(X_raw, y_raw, g_raw)

	# Initialize model with appropriate parameters
	if model_label == 'bayes_softmax3':
		model = ModelCls(draws=draws, tune=tune, chains=chains, target_accept=target_accept, random_seed=seed, cores=cores)
	else:
		model = ModelCls(random_seed=seed)

	model.fit(X_train, y_train, groups=g_train)

	# 2. VALIDATION: Use cross-validation on the validation file to update the model
	print(f'Phase 2: Updating {model_tag} on {val_file}')
	X_val, y_val, g_val = load_whole_dataset(val_file, prep_cfg)

	kf = KFold(n_splits=5, shuffle=True, random_state=seed)
	val_mses = []
	for i, (_, v_idx) in enumerate(kf.split(X_val)):
		# Predict on the "untouched" validation slice
		y_pred_fold = model.predict(X_val[v_idx], groups=g_val[v_idx])
		mse_fold = np.mean((y_val[v_idx] - y_pred_fold) ** 2)
		val_mses.append(mse_fold)
		print(f'  Fold {i + 1} Validation MSE: {mse_fold:.4f}')

	avg_mse = np.mean(val_mses)
	print(f'Average Validation MSE: {avg_mse:.4f}')

	# Save the balanced model now that it's validated
	model.save(paths, extra_meta={'train_src': train_file, 'val_src': val_file, 'avg_val_mse': avg_mse})

	# 3. TESTING: Purely unseen data
	print(f'Phase 3: Testing {model_tag} on {test_file}')
	X_test, y_test, g_test = load_whole_dataset(test_file, prep_cfg)

	# Calculate metrics
	test_metrics = model_graph_functions.evaluate_and_graph_clf(model, X_test, y_test, groups=g_test, name=f'{model_tag}_custom_pipeline', graph=True)

	# Generate confusion matrix if applicable
	if hasattr(model, 'predict_class'):
		y_pred_classes = model.predict_class(X_test, groups=g_test)
		model_graph_functions.plot_confusion_matrix(y_true=y_test, y_pred=y_pred_classes, name=f'{model_tag} Custom Pipeline Confusion Matrix')

	return {'test_metrics': test_metrics, 'model_type': model_tag, 'avg_val_mse': avg_mse}


def train_eval(
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
	target_accept: float = 0.99,
	seed: int = 123,
	cores: int = 8,
) -> Dict[str, Any]:
	if prep_cfg is None:
		prep_cfg = data_preparation.PrepConfig(dataset_name='unused')

	# 1. Setup Model Types
	ModelCls, model_tag = _select_model(model_label)
	paths = _model_paths(models_dir, train_base, model_tag)

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
	# Pass the existing model to be updated with validation data
	model = train_with_cross_val(val_base, model_label, prep_cfg, existing_model=model)

	# Save immediately after updating
	model.save(paths, extra_meta={'train_src': train_base, 'val_src': val_base})

	# 4. PHASE 3: TESTING (Evaluating on unseen data)
	print(f'--- Phase 3: Final Testing {model_tag} on {test_base} ---')
	X_test, y_test, groups_test = load_whole_dataset(test_base, prep_cfg)

	test_metrics = model_graph_functions.evaluate_and_graph_clf(
		model, X_test, y_test, groups=groups_test, name=f'{model_tag}_test_{test_base}', graph=True
	)

	# Create graph paths based on the test dataset name
	test_graph_path = paths['dir'] / f'{model_tag}_test_{test_base}.png'
	test_cm_path = paths['dir'] / f'{model_tag}_test_{test_base}_confusion.png'

	if test_graph_path:
		plt.savefig(test_graph_path)
		plt.close()
		print(f'Saved test graph to {test_graph_path}')

	# Confusion Matrix
	y_pred_cm = model.predict_class(X_test, groups=groups_test)
	model_graph_functions.plot_confusion_matrix(
		y_true=y_test, y_pred=y_pred_cm, name=f'{model_tag} Test Confusion Matrix - {test_base}', save_path=test_cm_path
	)
	print(f'Saved confusion matrix to {test_cm_path}')

	return {
		'trained': trained,
		'test_metrics': test_metrics,
		'paths': {'meta': str(paths['meta']), 'graph_test': str(test_graph_path), 'graph_cm': str(test_cm_path), 'model_dir': str(paths['dir'])},
	}


def test_on_new_data(
	test_base: str, model_label: str, *, prep_cfg: Optional[data_preparation.PrepConfig] = None, models_dir: str | Path = 'models'
) -> Dict[str, Any]:
	"""
	Loads a new dataset, prepares it using the model's original scaling params,
	and returns predictions and metrics without any training.
	"""
	if prep_cfg is None:
		prep_cfg = data_preparation.PrepConfig(dataset_name='unused')

	# 1. Setup Model Types and Paths
	ModelCls, model_tag = _select_model(model_label)
	# Fix: Need to pass a train_base for path construction
	paths = _model_paths(models_dir, 'default', model_tag)

	# 2. Check if Model Exists
	exists_check = paths['meta'].exists()
	if model_label == 'bayes_softmax3':
		exists_check = exists_check and paths['idata'].exists()

	if not exists_check:
		raise FileNotFoundError(f'Model not found. Expected files at {paths["meta"]}. Please train the model first using train_eval().')

	# 3. Load the Pre-trained Model
	print(f'Loading existing {model_tag} from {paths["meta"]}')
	model = ModelCls.load(paths)

	# 4. Prepare Test Data
	print(f'--- Testing {model_tag} on {test_base} ---')
	X_test, y_test, groups_test = load_whole_dataset(test_base, prep_cfg)

	# 5. Predict and Evaluate
	test_metrics = model_graph_functions.evaluate_and_graph_clf(
		model, X_test, y_test, groups=groups_test, name=f'{model_tag}_test_{test_base}', graph=True
	)

	# 6. Save Graphs
	# Create paths for this specific test evaluation
	test_graph_path = paths['dir'] / f'{model_tag}_test_{test_base}.png'
	test_cm_path = paths['dir'] / f'{model_tag}_test_{test_base}_confusion.png'

	if test_graph_path:
		plt.savefig(test_graph_path)
		plt.close()
		print(f'Saved test graph to {test_graph_path}')

	# Confusion Matrix
	y_pred_cm = model.predict_class(X_test, groups=groups_test)
	model_graph_functions.plot_confusion_matrix(
		y_true=y_test, y_pred=y_pred_cm, name=f'{model_tag} Test Confusion Matrix - {test_base}', save_path=test_cm_path
	)
	print(f'Saved confusion matrix to {test_cm_path}')

	return {
		'test_metrics': test_metrics,
		'paths': {'graph_test': str(test_graph_path), 'graph_cm': str(test_cm_path), 'model_dir': str(paths['dir'])},
	}


def train_eval_all(train_f, val_f, test_f):
	"""Runs both models for the capstone comparison."""
	results = {}
	for label in ['bayes_softmax3', 'multi_log_regression']:
		results[label] = train_eval(train_f, val_f, test_f, label)
	return results


if __name__ == '__main__':
	# train_eval_all('testing.training', 'testing.validation', 'testing.testing')
	train_eval_all('bettersample.training', 'bettersample.validation', 'bettersample.testing')
	# print(test_on_new_data('testing.training', 'testing.testing', 'bayes_softmax3'))
