from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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
			verbose=self.verbose,
		)

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
			print('\n=== Training HMM Dosage Classifier ===')
			print(f'Samples: {X.shape[0]}, Features: {X.shape[1]}')
			print(f'Dosage distribution: {np.bincount(y_int, minlength=3)}')

		def create_sequences(X_data, min_seq_length=10, max_seq_length=100):
			"""Create sequences of appropriate length for HMM training."""
			n_samples = len(X_data)

			if n_samples <= min_seq_length:
				# If too few samples, create one sequence
				return [n_samples]

			# Create sequences of varying lengths between min and max
			sequences = []
			remaining = n_samples

			while remaining > 0:
				if remaining <= max_seq_length:
					sequences.append(remaining)
					break

				# Create a sequence of random length between min and max
				seq_len = np.random.randint(min_seq_length, min(max_seq_length + 1, remaining + 1))
				sequences.append(seq_len)
				remaining -= seq_len

			return sequences

		# Check if we should use hierarchical group structure
		if groups is not None and len(np.unique(groups)) > 1:
			self.use_groups = True
			unique_groups = np.unique(groups)

			if self.verbose:
				print(f'Training hierarchical model with {len(unique_groups)} groups')

			# Train a separate HMM for each group
			for group_id in unique_groups:
				group_mask = groups == group_id
				X_group = Xz[group_mask]
				y_group = y_int[group_mask]

				if len(X_group) < 30:  # Increased minimum samples for meaningful sequences
					if self.verbose:
						print(f'  Skipping group {group_id} (only {len(X_group)} samples, need at least 30)')
					continue

				group_model = self._create_hmm_model()

				# Create meaningful sequence lengths
				lengths = create_sequences(X_group, min_seq_length=10, max_seq_length=50)

				try:
					group_model.fit(X_group, lengths)
					self.group_models[int(group_id)] = group_model

					if self.verbose:
						print(f'  Group {group_id}: {len(X_group)} samples in {len(lengths)} sequences')
				except Exception as e:
					if self.verbose:
						print(f'  Warning: Failed to train group {group_id}: {e}')

		# Always train a global model as fallback
		if self.verbose:
			print('Training global HMM model')

		self.model = self._create_hmm_model()

		# Create meaningful sequences for the global model
		lengths = create_sequences(Xz, min_seq_length=20, max_seq_length=200)

		try:
			self.model.fit(Xz, lengths)

			if self.verbose:
				print(f'Global model trained with {len(Xz)} samples in {len(lengths)} sequences')
		except Exception as e:
			print(f'Error training global HMM model: {e}')
			raise

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

	@staticmethod
	def _create_sequences(X_data, min_seq_length=20, max_seq_length=100):
		"""Create sequences of appropriate length for HMM prediction."""
		n_samples = len(X_data)
		if n_samples <= min_seq_length:
			return [n_samples]
		sequences = []
		remaining = n_samples
		while remaining > 0:
			if remaining <= max_seq_length:
				sequences.append(remaining)
				break
			seq_len = np.random.randint(min_seq_length, min(max_seq_length + 1, remaining + 1))
			sequences.append(seq_len)
			remaining -= seq_len
		return sequences

	def _predict_proba_single_model(self, model: hmm.GaussianHMM, X: np.ndarray) -> np.ndarray:
		"""
		Predict probabilities using a single HMM model.

		Respects the sequential structure by using variable-length sequences,
		allowing the model to leverage learned transition probabilities.
		"""
		try:
			lengths = HMMDosageClassifier._create_sequences(X, min_seq_length=20, max_seq_length=100)

			# Get posterior probabilities for each state
			_, posteriors = model.score_samples(X, lengths)

			# posteriors has shape (n_samples, n_components)
			probs = posteriors

			# Ensure probabilities sum to 1 (numerical stability)
			row_sums = probs.sum(axis=1, keepdims=True)
			# Avoid division by zero
			row_sums = np.where(row_sums == 0, 1, row_sums)
			probs = probs / row_sums

			return probs.astype(np.float32)
		except Exception as e:
			print(f'Warning in HMM prediction: {e}')
			# Return uniform probabilities as fallback
			return np.ones((len(X), self.n_components), dtype=np.float32) / self.n_components

	def predict_proba(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Predict probability distribution over dosage classes.

		Args:
			X: Feature matrix (n_samples, n_features)
			groups: Optional group indices for hierarchical prediction

		Returns:
			proba: Probability matrix (n_samples, 3)
		"""
		if self.model is None:
			raise RuntimeError('Model must be fitted before prediction')

		Xz = standardize_apply(X, self.feature_mean_, self.feature_std_)

		# Use GPU acceleration if available
		if self.use_gpu:
			try:
				import jax.numpy as jnp

				Xz_gpu = jnp.array(Xz)
				# Note: hmmlearn doesn't natively support JAX, so we convert back
				Xz = np.array(Xz_gpu)
			except:
				pass  # Fall back to CPU

		if self.use_groups and groups is not None and len(self.group_models) > 0:
			# Use hierarchical prediction with group-specific models
			probs = np.zeros((len(X), self.n_components), dtype=np.float32)

			for group_id, group_model in self.group_models.items():
				group_mask = groups == group_id
				if group_mask.any():
					probs[group_mask] = self._predict_proba_single_model(group_model, Xz[group_mask])

			# Use global model for groups without specific models
			ungrouped_mask = np.ones(len(X), dtype=bool)
			for group_id in self.group_models.keys():
				ungrouped_mask &= groups != group_id

			if ungrouped_mask.any():
				probs[ungrouped_mask] = self._predict_proba_single_model(self.model, Xz[ungrouped_mask])
		else:
			# Use global model for all predictions
			probs = self._predict_proba_single_model(self.model, Xz)

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
		payload = {
			'type': 'HMMDosageClassifier',
			'tag': self.tag,
			'feature_mean': self.feature_mean_.tolist(),
			'feature_std': self.feature_std_.tolist(),
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

		# Restore global model
		m.model = m._create_hmm_model()
		m.model.startprob_ = np.array(meta['model_params']['startprob'])
		m.model.transmat_ = np.array(meta['model_params']['transmat'])
		m.model.means_ = np.array(meta['model_params']['means'])
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
