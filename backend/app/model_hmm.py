from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from hmmlearn import hmm
from pycm import ConfusionMatrix

from .model_functions import coerce_dosage_classes, ensure_dir, load_meta, save_common_meta, standardize_apply, standardize_fit

# Configure GPU acceleration for HMM if available
try:
	import os

	# Configure for GPU compatibility
	os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
	os.environ.setdefault('JAX_ENABLE_X64', 'false')

	import jax
	import jax.numpy as jnp

	# Configure JAX for optimal performance
	jax.config.update('jax_enable_x64', False)  # Use float32 for speed

	# Memory optimization
	os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.8')
	os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

	# Check if CUDA is available
	if jax.devices('gpu'):
		print(f'GPU devices found for HMM: {jax.devices("gpu")}')
		jax.config.update('jax_platform_name', 'gpu')
		os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
		GPU_AVAILABLE = True
	else:
		print('No GPU devices found for HMM, using CPU')
		jax.config.update('jax_platform_name', 'cpu')
		GPU_AVAILABLE = False
except ImportError:
	print('JAX not installed, HMM will use CPU backend')
	GPU_AVAILABLE = False
except Exception as e:
	print(f'GPU setup warning for HMM: {e}')
	GPU_AVAILABLE = False


class HMMDosageClassifier:
	"""
	Hidden Markov Model for genetic dosage classification using hmmlearn.

	This model treats the sequential nature of genetic variants, where each
	observation is a feature vector and the hidden states correspond to dosage
	classes (0, 1, 2).

	Features:
	- Uses Gaussian emissions for continuous features
	- Supports hierarchical group structure via group-specific training
	- GPU acceleration support via JAX backend
	- Comprehensive metrics via PYCM (Python Confusion Matrix)
	- Compatible with existing model_functions utilities

	Returns:
	  - predict_proba(X): (n, 3) probability distribution over dosage classes
	  - predict_class(X): (n,) most likely dosage class
	  - predict(X): expected dosage E[y] for regression metric compatibility
	"""

	def __init__(
		self,
		*,
		n_iter: int = 100,
		n_components: int = 3,  # 3 hidden states for dosage 0, 1, 2
		covariance_type: str = 'diag',  # 'diag', 'full', 'spherical', 'tied'
		random_seed: int = 123,
		tol: float = 1e-2,
		use_gpu: bool = True,
		verbose: bool = True,
		n_mix: int = 1,  # Number of Gaussian mixtures per state
	):
		"""
		Initialize HMM Dosage Classifier.

		Args:
			n_iter: Maximum number of EM iterations
			n_components: Number of hidden states (fixed to 3 for dosages 0, 1, 2)
			covariance_type: Type of covariance parameters
			random_seed: Random seed for reproducibility
			tol: Convergence threshold
			use_gpu: Whether to use GPU acceleration (if available)
			verbose: Print training progress
			n_mix: Number of Gaussian mixture components per state
		"""
		self.n_iter = n_iter
		self.n_components = n_components
		self.covariance_type = covariance_type
		self.random_seed = random_seed
		self.tol = tol
		self.use_gpu = use_gpu and GPU_AVAILABLE
		self.verbose = verbose
		self.n_mix = n_mix

		# Model components
		self.model: Optional[hmm.GaussianHMM] = None
		self.feature_mean_: Optional[np.ndarray] = None
		self.feature_std_: Optional[np.ndarray] = None
		self.state_to_dosage_: Optional[Dict[int, int]] = None  # Mapping from HMM states to dosage classes

		# Group-specific models for hierarchical structure
		self.group_models: Dict[int, hmm.GaussianHMM] = {}
		self.use_groups: bool = False

		# PYCM confusion matrix for comprehensive metrics
		self.pycm_train_: Optional[ConfusionMatrix] = None
		self.pycm_metrics_: Optional[Dict[str, Any]] = None

		if self.use_gpu:
			print('🚀 HMM GPU acceleration enabled')
		else:
			print('💻 HMM running on CPU')

	@property
	def tag(self) -> str:
		return 'hmm_dosage'

	def _create_hmm_model(self) -> hmm.GaussianHMM:
		"""Create a new HMM model instance with current parameters."""
		return hmm.GaussianHMM(
			n_components=self.n_components,
			covariance_type=self.covariance_type,
			n_iter=self.n_iter,
			tol=self.tol,
			random_state=self.random_seed,
			verbose=False,  # Suppress verbose output for cleaner logs
			init_params='',  # Don't initialize parameters - we'll do it manually
			params='stmc',  # But do update all parameters during training
		)

	def _initialize_hmm_with_priors(self, model: hmm.GaussianHMM, X: np.ndarray, y: np.ndarray, n_features_out: int = 1) -> None:
		"""Initialize HMM parameters using label information (semi-supervised approach).

		Args:
			model: HMM model to initialize
			X: Original feature matrix (before reshaping)
			y: Labels
			n_features_out: Number of features in reshaped data (1 for our sequence approach)
		"""
		# Set n_features manually so we can initialize parameters
		model.n_features = n_features_out

		# Initialize means array
		model.means_ = np.zeros((self.n_components, n_features_out))  # (n_states, n_dims)

		# For each dosage class, compute the mean of all features and use as initial state mean
		for dosage in range(3):
			class_mask = y == dosage
			if class_mask.sum() > 0:
				# Compute mean of all features for samples in this class
				class_mean = X[class_mask].mean()  # Mean across all features and samples
				model.means_[dosage, 0] = class_mean
			else:
				# If no samples, use neutral initialization
				model.means_[dosage, 0] = 0.0

				# Initialize covariances based on class variance
		# Compute variance for each class
		class_vars = []
		for dosage in range(3):
			class_mask = y == dosage
			if class_mask.sum() > 1:
				class_var = X[class_mask].var()
				class_vars.append(max(class_var, 0.01))  # Avoid zero variance
			else:
				class_vars.append(1.0)

		# Initialize covariances based on covariance_type
		if self.covariance_type == 'diag':
			model.covars_ = np.array([[v] for v in class_vars])  # (n_states, n_features)
		elif self.covariance_type == 'spherical':
			model.covars_ = np.array(class_vars)  # (n_states,)
		elif self.covariance_type == 'full':
			model.covars_ = np.array([[[v]] for v in class_vars])  # (n_states, n_features, n_features)
		elif self.covariance_type == 'tied':
			# Use average variance across all classes
			avg_var = np.mean(class_vars)
			model.covars_ = np.array([[avg_var]])  # (n_features, n_features)

		# Initialize transition matrix to favor staying in same state
		# This reflects that dosage tends to be consistent within regions
		model.transmat_ = np.array(
			[
				[0.85, 0.10, 0.05],  # From state 0: likely stay in 0
				[0.10, 0.80, 0.10],  # From state 1: likely stay in 1
				[0.05, 0.10, 0.85],  # From state 2: likely stay in 2
			]
		)

		# Initialize start probabilities based on class distribution
		class_counts = np.bincount(y.astype(int), minlength=3)
		model.startprob_ = (class_counts + 1) / (class_counts.sum() + 3)  # Laplace smoothing

		if self.verbose:
			print('Initialized HMM with label-guided priors')
			print(f'  Start probabilities: {model.startprob_}')
			print(f'  Means: {model.means_.flatten()}')
			print(f'  Transition matrix diagonal: {np.diag(model.transmat_)}')

	def _align_states_to_dosages(self, X: np.ndarray, y: np.ndarray, lengths: list) -> Dict[int, int]:
		"""Align HMM hidden states to actual dosage classes using training labels."""
		# Get state assignments using Viterbi algorithm
		state_sequence = self.model.predict(X, lengths)

		# For each state, find which dosage it corresponds to
		state_to_dosage = {}
		state_dosage_counts = {}  # Track distribution for diagnostics

		for state in range(self.n_components):
			state_mask = state_sequence == state
			if state_mask.sum() > 0:
				# Find dosage distribution for this state
				dosages_in_state = y[state_mask]
				dosage_counts = np.bincount(dosages_in_state.astype(int), minlength=3)
				state_dosage_counts[state] = dosage_counts

				# Assign state to most common dosage
				most_common = dosage_counts.argmax()
				state_to_dosage[state] = most_common

				if self.verbose:
					purity = dosage_counts[most_common] / dosage_counts.sum() * 100
					print(f'  State {state} -> Dosage {most_common} ({state_mask.sum()} obs, {purity:.1f}% purity)')
					print(f'    Distribution: D0={dosage_counts[0]}, D1={dosage_counts[1]}, D2={dosage_counts[2]}')
			else:
				# If no observations in this state, assign to missing dosage
				used_dosages = set(state_to_dosage.values())
				for d in range(3):
					if d not in used_dosages:
						state_to_dosage[state] = d
						if self.verbose:
							print(f'  State {state} -> Dosage {d} (no observations, assigned to missing class)')
						break

				# Check if all dosages are covered
		assigned_dosages = set(state_to_dosage.values())
		if len(assigned_dosages) < 3:
			if self.verbose:
				print(f'  WARNING: Not all dosages assigned! Missing: {set(range(3)) - assigned_dosages}')
				print('  Attempting to reassign states based on means...')

			# Sort states by their learned means (should correlate with dosage)
			state_means = self.model.means_.flatten()
			sorted_states = np.argsort(state_means)

			# Assign sorted states to dosages 0, 1, 2
			state_to_dosage = {int(sorted_states[i]): i for i in range(3)}

			if self.verbose:
				print('  Reassigned based on means:')
				for state, dosage in sorted(state_to_dosage.items()):
					print(f'    State {state} (mean={state_means[state]:.3f}) -> Dosage {dosage}')

		return state_to_dosage

	def _reshape_for_sequence(self, X: np.ndarray) -> Tuple[np.ndarray, list]:
		"""Treat each sample's features as a temporal sequence.

		Instead of treating samples as arbitrary sequences, treat each sample's
		features (which represent positions along genome) as a sequence.

		Args:
			X: Feature matrix (n_samples, n_features)

		Returns:
			X_seq: Reshaped data (total_features, 1) where each feature is a timestep
			lengths: List of sequence lengths (all equal to n_features)
		"""
		n_samples, n_features = X.shape

		# Each sample becomes a sequence of length n_features
		# Each timestep is a 1D observation (single feature value)
		X_seq = X.reshape(-1, 1)  # Flatten all samples into (n_samples * n_features, 1)
		lengths = [n_features] * n_samples  # Each sample is a sequence of n_features

		return X_seq, lengths

	def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> 'HMMDosageClassifier':
		"""
		Fit HMM to genetic dosage data.

		Args:
			X: Feature matrix (n_samples, n_features)
			y: Dosage labels (n_samples,) with values in {0, 1, 2}
			groups: Optional group indices for hierarchical modeling

		Returns:
			self: Fitted model
		"""
		X = np.asarray(X, dtype=np.float32)
		y_int = coerce_dosage_classes(np.asarray(y, dtype=np.float32))

		# Standardize features
		Xz, mu, sd = standardize_fit(X)
		self.feature_mean_ = mu
		self.feature_std_ = sd

		if self.verbose:
			print('\n=== Training HMM Dosage Classifier (Semi-Supervised) ===')
			print(f'Samples: {X.shape[0]}, Features: {X.shape[1]}')
			print(f'Dosage distribution: {np.bincount(y_int, minlength=3)}')
			print('Using within-sample sequential structure (features as time steps)')

		# Reshape data: treat each sample's features as a sequence
		X_seq, lengths = self._reshape_for_sequence(Xz)

		# Create and flatten y for alignment (repeat each label n_features times)
		n_features = X.shape[1]
		y_seq = np.repeat(y_int, n_features)

		if self.verbose:
			print(f'Reshaped to {len(lengths)} sequences of length {n_features} each')
			print(f'Total observations: {X_seq.shape[0]}')

			# Create and initialize model
		self.model = self._create_hmm_model()

		# Initialize with label-guided priors (semi-supervised)
		# Note: n_features_out=1 because we reshape each feature to a 1D observation
		self._initialize_hmm_with_priors(self.model, Xz, y_int, n_features_out=1)

		try:
			# Train HMM on sequences
			self.model.fit(X_seq, lengths)

			if self.verbose:
				print('\nHMM training complete')
				print('Learned transition matrix:')
				for i in range(3):
					print(f'  State {i}: {self.model.transmat_[i]}')
		except Exception as e:
			print(f'Error training HMM model: {e}')
			raise

			# Align HMM states to dosage classes using training labels
		if self.verbose:
			print('\nAligning HMM states to dosage classes...')
		state_alignment = self._align_states_to_dosages(X_seq, y_seq, lengths)
		# Convert numpy int64 to Python int for JSON serialization
		self.state_to_dosage_ = {int(k): int(v) for k, v in state_alignment.items()}

		# Generate training predictions and compute PYCM metrics
		if self.verbose:
			print('\nComputing training metrics with PYCM...')

		y_pred_train = self.predict_class(X, groups=groups)
		self.pycm_train_ = ConfusionMatrix(actual_vector=y_int.tolist(), predict_vector=y_pred_train.tolist(), digit=4)

		# Helper function to safely convert PYCM metrics to float
		def safe_metric(value):
			"""Convert PYCM metric to float, handling 'None' strings and None values."""
			if value is None or value == 'None':
				return 0.0
			try:
				return float(value)
			except (ValueError, TypeError):
				return 0.0

		# Store key metrics for later access
		self.pycm_metrics_ = {
			'overall_accuracy': safe_metric(self.pycm_train_.Overall_ACC),
			'kappa': safe_metric(self.pycm_train_.Kappa),
			'overall_f1': safe_metric(self.pycm_train_.F1_Macro),
			'overall_precision': safe_metric(self.pycm_train_.PPV_Macro),
			'overall_recall': safe_metric(self.pycm_train_.TPR_Macro),
			'class_accuracy': {k: safe_metric(v) for k, v in self.pycm_train_.ACC.items()},
			'class_f1': {k: safe_metric(v) for k, v in self.pycm_train_.F1.items()},
			'class_precision': {k: safe_metric(v) for k, v in self.pycm_train_.PPV.items()},
			'class_recall': {k: safe_metric(v) for k, v in self.pycm_train_.TPR.items()},
		}

		if self.verbose:
			print('\n=== Training Metrics (PYCM) ===')
			print(f'Overall Accuracy: {self.pycm_metrics_["overall_accuracy"]:.4f}')
			print(f'Kappa: {self.pycm_metrics_["kappa"]:.4f}')
			print(f'Macro F1-Score: {self.pycm_metrics_["overall_f1"]:.4f}')
			print(f'Macro Precision: {self.pycm_metrics_["overall_precision"]:.4f}')
			print(f'Macro Recall: {self.pycm_metrics_["overall_recall"]:.4f}')
			print('\nPer-Class F1-Scores:')
			for cls, f1 in self.pycm_metrics_['class_f1'].items():
				print(f'  Dosage {cls}: {f1:.4f}')

		return self

	def _predict_sample_dosage(self, X_sample: np.ndarray) -> np.ndarray:
		"""Predict dosage for a single sample using posterior probabilities.

		Args:
			X_sample: Single sample's features (n_features,)

		Returns:
			probabilities: Probability distribution over dosages (3,)
		"""
		# Reshape sample to sequence format
		X_seq = X_sample.reshape(-1, 1)  # (n_features, 1)
		n_features = len(X_sample)

		# Use forward-backward algorithm to get posterior probabilities for each state
		# This gives soft probabilities rather than hard Viterbi assignments
		_, state_posteriors = self.model.score_samples(X_seq, [n_features])
		# state_posteriors shape: (n_features, n_states)

		# Map state probabilities to dosage probabilities
		dosage_probs = np.zeros(3, dtype=np.float32)
		for state in range(self.n_components):
			if state not in self.state_to_dosage_:
				# Skip states not in mapping (shouldn't happen, but defensive)
				continue
			dosage = self.state_to_dosage_[state]
			# Sum probabilities for this state across all timesteps
			dosage_probs[dosage] += state_posteriors[:, state].sum()

			# Normalize to get probability distribution
		prob_sum = dosage_probs.sum()
		if prob_sum > 0:
			dosage_probs = dosage_probs / prob_sum
		else:
			# Fallback to uniform if something went wrong
			dosage_probs = np.ones(3, dtype=np.float32) / 3.0

		return dosage_probs

	def _predict_proba_with_posteriors(self, X: np.ndarray) -> np.ndarray:
		"""
		Predict probabilities using posterior probabilities from forward-backward algorithm.

		For each sample:
		  1. Treat its features as a sequence
		  2. Use forward-backward to get state posteriors
		  3. Map state probabilities to dosages using learned alignment
		  4. Aggregate across timesteps to get dosage probabilities

		Args:
			X: Feature matrix (n_samples, n_features)

		Returns:
			probs: Probability matrix (n_samples, 3)
		"""
		try:
			n_samples = X.shape[0]
			probs = np.zeros((n_samples, 3), dtype=np.float32)

			# Predict each sample independently
			for i in range(n_samples):
				probs[i] = self._predict_sample_dosage(X[i])

			return probs
		except Exception as e:
			print(f'Warning in HMM prediction: {e}')
			import traceback

			traceback.print_exc()
			# Return uniform probabilities as fallback
			return np.ones((len(X), 3), dtype=np.float32) / 3.0

	def predict_proba(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Predict probability distribution over dosage classes using posterior probabilities.

		Args:
			X: Feature matrix (n_samples, n_features)
			groups: Optional group indices (not used in current implementation)

		Returns:
			proba: Probability matrix (n_samples, 3)
		"""
		if self.model is None:
			raise RuntimeError('Model must be fitted before prediction')

		if self.state_to_dosage_ is None or len(self.state_to_dosage_) == 0:
			raise RuntimeError('State alignment not computed. Model may not be properly trained.')

		Xz = standardize_apply(X, self.feature_mean_, self.feature_std_)

		# Use posterior probabilities (forward-backward) for soft predictions
		probs = self._predict_proba_with_posteriors(Xz)

		return probs

	def predict_class(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Predict most likely dosage class.

		Args:
			X: Feature matrix (n_samples, n_features)
			groups: Optional group indices

		Returns:
			classes: Predicted dosage classes (n_samples,)
		"""
		probs = self.predict_proba(X, groups=groups)
		return np.argmax(probs, axis=1).astype(np.int64)

	def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Expected dosage E[y] = sum_c c * p(c).
		This makes it compatible with regression metrics/plots.

		Args:
			X: Feature matrix (n_samples, n_features)
			groups: Optional group indices

		Returns:
			expected_dosage: Expected dosage values (n_samples,)
		"""
		probs = self.predict_proba(X, groups=groups)
		classes = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		return (probs * classes[None, :]).sum(axis=1)

	def evaluate_pycm(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> ConfusionMatrix:
		"""
		Evaluate model on test data using PYCM for comprehensive metrics.

		Args:
			X: Feature matrix
			y: True dosage labels
			groups: Optional group indices

		Returns:
			pycm_cm: PYCM ConfusionMatrix object with detailed metrics
		"""
		y_int = coerce_dosage_classes(np.asarray(y, dtype=np.float32))
		y_pred = self.predict_class(X, groups=groups)

		pycm_cm = ConfusionMatrix(actual_vector=y_int.tolist(), predict_vector=y_pred.tolist(), digit=4)

		return pycm_cm

	def print_pycm_report(self, pycm_cm: ConfusionMatrix, title: str = 'HMM Evaluation Report'):
		"""
		Print a comprehensive PYCM evaluation report.

		Args:
			pycm_cm: PYCM ConfusionMatrix object
			title: Report title
		"""

		# Helper function to safely format PYCM metrics
		def safe_format(value):
			"""Convert PYCM metric to float for formatting, handling 'None' strings."""
			if value is None or value == 'None':
				return 0.0
			try:
				return float(value)
			except (ValueError, TypeError):
				return 0.0

		print(f'\n{"=" * 60}')
		print(f'{title:^60}')
		print(f'{"=" * 60}')

		print('\n=== Overall Metrics ===')
		print(f'Overall Accuracy: {safe_format(pycm_cm.Overall_ACC):.4f}')
		print(f'Balanced Accuracy: {safe_format(pycm_cm.TPR_Macro):.4f}')
		print(f'Kappa: {safe_format(pycm_cm.Kappa):.4f}')
		print(f'F1-Score (Macro): {safe_format(pycm_cm.F1_Macro):.4f}')
		print(f'Precision (Macro): {safe_format(pycm_cm.PPV_Macro):.4f}')
		print(f'Recall (Macro): {safe_format(pycm_cm.TPR_Macro):.4f}')

		print('\n=== Per-Class Metrics ===')
		print(f'{"Class":<10} {"F1":<10} {"Precision":<12} {"Recall":<10} {"Accuracy":<10}')
		print('-' * 60)
		for cls in sorted(pycm_cm.classes):
			f1 = safe_format(pycm_cm.F1[cls]) if cls in pycm_cm.F1 else 0.0
			prec = safe_format(pycm_cm.PPV[cls]) if cls in pycm_cm.PPV else 0.0
			rec = safe_format(pycm_cm.TPR[cls]) if cls in pycm_cm.TPR else 0.0
			acc = safe_format(pycm_cm.ACC[cls]) if cls in pycm_cm.ACC else 0.0
			print(f'Dosage {cls:<3} {f1:<10.4f} {prec:<12.4f} {rec:<10.4f} {acc:<10.4f}')

		print('\n=== Confusion Matrix ===')
		print(pycm_cm)
		print('=' * 60)

	def save(self, paths: Dict[str, Path], extra_meta: Dict[str, Any]) -> None:
		"""
		Save HMM model to disk.

		Args:
			paths: Dictionary with file paths (dir, meta, etc.)
			extra_meta: Additional metadata to save
		"""
		if self.model is None:
			raise RuntimeError('No model to save.')

		ensure_dir(paths['dir'])

		# Save model parameters
		# Ensure state_to_dosage uses Python ints, not numpy int64
		state_to_dosage_serializable = {int(k): int(v) for k, v in self.state_to_dosage_.items()}

		payload = {
			'type': 'HMMDosageClassifier',
			'tag': self.tag,
			'feature_mean': self.feature_mean_.tolist(),
			'feature_std': self.feature_std_.tolist(),
			'state_to_dosage': state_to_dosage_serializable,
			'params': {
				'n_iter': self.n_iter,
				'n_components': self.n_components,
				'covariance_type': self.covariance_type,
				'random_seed': self.random_seed,
				'tol': self.tol,
				'use_gpu': self.use_gpu,
				'verbose': self.verbose,
				'n_mix': self.n_mix,
			},
			'model_params': {
				'startprob': self.model.startprob_.tolist(),
				'transmat': self.model.transmat_.tolist(),
				'means': self.model.means_.tolist(),
				'covars': self.model.covars_.tolist(),
			},
			'use_groups': self.use_groups,
			'n_group_models': len(self.group_models),
			'pycm_metrics': self.pycm_metrics_,
			'extra': extra_meta,
		}

		# Save group models if they exist
		if self.group_models:
			group_params = {}
			for group_id, gmodel in self.group_models.items():
				group_params[int(group_id)] = {
					'startprob': gmodel.startprob_.tolist(),
					'transmat': gmodel.transmat_.tolist(),
					'means': gmodel.means_.tolist(),
					'covars': gmodel.covars_.tolist(),
				}
			payload['group_models'] = group_params

		save_common_meta(paths, payload)

		# Save PYCM confusion matrix if available
		if self.pycm_train_ is not None:
			pycm_path = paths['dir'] / f'{paths["meta"].stem}.pycm.obj'
			self.pycm_train_.save_obj(str(pycm_path))

	@classmethod
	def load(cls, paths: Dict[str, Path]) -> 'HMMDosageClassifier':
		"""
		Load HMM model from disk.

		Args:
			paths: Dictionary with file paths

		Returns:
			model: Loaded HMMDosageClassifier
		"""
		meta = load_meta(paths)

		# Create model instance
		m = cls(**meta['params'])

		# Restore feature standardization parameters
		m.feature_mean_ = np.array(meta['feature_mean'], dtype=np.float32)
		m.feature_std_ = np.array(meta['feature_std'], dtype=np.float32)
		m.state_to_dosage_ = meta.get('state_to_dosage', {0: 0, 1: 1, 2: 2})
		# Convert string keys back to int if necessary
		if isinstance(list(m.state_to_dosage_.keys())[0], str):
			m.state_to_dosage_ = {int(k): int(v) for k, v in m.state_to_dosage_.items()}

			# Restore global model
		m.model = m._create_hmm_model()
		# Set n_features before restoring parameters
		means_array = np.array(meta['model_params']['means'])
		m.model.n_features = means_array.shape[1]  # Get n_features from means shape
		m.model.startprob_ = np.array(meta['model_params']['startprob'])
		m.model.transmat_ = np.array(meta['model_params']['transmat'])
		m.model.means_ = means_array
		m.model.covars_ = np.array(meta['model_params']['covars'])

		# Restore group models if they exist
		m.use_groups = meta.get('use_groups', False)
		if 'group_models' in meta:
			for group_id_str, gparams in meta['group_models'].items():
				group_id = int(group_id_str)
				gmodel = m._create_hmm_model()
				gmodel.startprob_ = np.array(gparams['startprob'])
				gmodel.transmat_ = np.array(gparams['transmat'])
				gmodel.means_ = np.array(gparams['means'])
				gmodel.covars_ = np.array(gparams['covars'])
				m.group_models[group_id] = gmodel

		# Restore PYCM metrics
		m.pycm_metrics_ = meta.get('pycm_metrics')

		# Try to load PYCM confusion matrix object
		pycm_path = paths['dir'] / f'{paths["meta"].stem}.pycm.obj'
		if pycm_path.exists():
			try:
				m.pycm_train_ = ConfusionMatrix(file=open(str(pycm_path)))
			except:
				pass  # OK if we can't load it

		return m
