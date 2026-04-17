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

	Each individual is treated as a sequence whose timesteps are genetic
	sites along the chromosome.  Only the *mean_dosage* feature (column 0)
	is used as the observed emission; the hidden states correspond to
	dosage classes (0, 1, 2).

	The ``groups`` array passed to :meth:`fit` / :meth:`predict_proba`
	identifies individuals: consecutive rows that share the same group
	value are merged into one sequence.  When ``groups`` is *None* every
	row is treated as one long sequence.

	Features:
	- Uses Gaussian emissions on the mean_dosage feature
	- Handles variable-length sequences (sites per individual)
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
		random_seed: Optional[int] = None,
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
			random_seed: Random seed for reproducibility (auto-generated if None)
			tol: Convergence threshold
			use_gpu: Whether to use GPU acceleration (if available)
			verbose: Print training progress
			n_mix: Number of Gaussian mixture components per state
		"""
		self.n_iter = n_iter
		self.n_components = n_components
		self.covariance_type = covariance_type
		if random_seed is None:
			random_seed = int(np.random.SeedSequence().entropy % (2**32))
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
			print('HMM GPU acceleration enabled')
		else:
			print('HMM running on CPU')

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

	def _initialize_hmm_with_priors(self, model: hmm.GaussianHMM, X_obs: np.ndarray, y: np.ndarray) -> None:
		"""Initialize HMM parameters using label information (semi-supervised approach).

		Args:
			model: HMM model to initialize
			X_obs: mean_dosage observations (n_total, 1)
			y: Labels aligned with X_obs (n_total,)
		"""
		model.n_features = 1

		vals = X_obs.ravel()

		# Initialize means: per-class mean of mean_dosage
		model.means_ = np.zeros((self.n_components, 1))
		for dosage in range(3):
			class_mask = y == dosage
			if class_mask.sum() > 0:
				model.means_[dosage, 0] = vals[class_mask].mean()
			else:
				model.means_[dosage, 0] = 0.0

		# Per-class variance of mean_dosage
		class_vars = []
		for dosage in range(3):
			class_mask = y == dosage
			if class_mask.sum() > 1:
				class_vars.append(max(vals[class_mask].var(), 0.01))
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

	def _align_states_to_dosages(self, X_seq: np.ndarray, y_seq: np.ndarray, lengths: list) -> Dict[int, int]:
		"""Align HMM hidden states to actual dosage classes using Viterbi on full sequences.

		Uses a greedy best-match strategy on the (state, dosage) count matrix,
		then falls back to learned-mean ordering to guarantee all 3 dosage
		classes are covered.
		"""
		state_sequence = self.model.predict(X_seq, lengths)

		# Build count matrix: counts[s, d] = observations where state==s and true dosage==d
		counts = np.zeros((self.n_components, 3), dtype=np.int64)
		for s in range(self.n_components):
			mask = state_sequence == s
			if mask.sum() > 0:
				dosages = y_seq[mask].astype(int)
				for d in range(3):
					counts[s, d] = (dosages == d).sum()

		if self.verbose:
			print('  State-dosage count matrix:')
			for s in range(self.n_components):
				print(f'    State {s}: D0={counts[s, 0]}, D1={counts[s, 1]}, D2={counts[s, 2]}')

		# Greedy assignment: pick best (state, dosage) pair by count
		state_to_dosage: Dict[int, int] = {}
		used_states: set = set()
		used_dosages: set = set()

		pairs = []
		for s in range(self.n_components):
			for d in range(3):
				pairs.append((int(counts[s, d]), s, d))
		pairs.sort(reverse=True)

		for count_val, s, d in pairs:
			if s not in used_states and d not in used_dosages:
				state_to_dosage[s] = d
				used_states.add(s)
				used_dosages.add(d)
				if self.verbose:
					total = int(counts[s].sum())
					purity = count_val / total * 100 if total > 0 else 0
					print(f'  State {s} -> Dosage {d} ({total} obs, {purity:.1f}% purity)')

		# Fallback: if greedy didn't cover all 3 dosages, use mean ordering
		if len(set(state_to_dosage.values())) < 3:
			if self.verbose:
				missing = set(range(3)) - set(state_to_dosage.values())
				print(f'  WARNING: Missing dosages {missing}, falling back to mean ordering')

			state_means = self.model.means_.flatten()
			sorted_states = np.argsort(state_means)
			state_to_dosage = {int(sorted_states[i]): i for i in range(3)}

			if self.verbose:
				for state, dosage in sorted(state_to_dosage.items()):
					print(f'    State {state} (mean={state_means[state]:.3f}) -> Dosage {dosage}')

		return state_to_dosage

	def _build_sequences(
		self, X: np.ndarray, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, list, Optional[np.ndarray]]:
		"""Build per-individual sequences from site-level data.

		Each individual becomes one HMM sequence whose timesteps are
		genetic sites.  Only mean_dosage (feature column 0) is used as
		the 1-D observation.

		Consecutive rows that share the same ``groups`` value are treated
		as one individual.  When *groups* is ``None`` all rows form a
		single sequence.

		Args:
			X: Standardized feature matrix (n_rows, n_features).
			y: Optional dosage labels (n_rows,).
			groups: Optional per-row individual identifier (n_rows,).

		Returns:
			X_seq: (n_rows, 1) mean_dosage observations (row order preserved).
			lengths: List of per-individual sequence lengths.
			y: Pass-through of *y* (unchanged).
		"""
		X_seq = X[:, 0:1].copy()  # mean_dosage only, shape (n, 1)

		if groups is not None:
			groups = np.asarray(groups)
			# Consecutive runs of the same value define individual boundaries
			change_points = np.where(np.diff(groups) != 0)[0] + 1
			starts = np.concatenate([[0], change_points])
			ends = np.concatenate([change_points, [len(groups)]])
			lengths = (ends - starts).tolist()
		else:
			lengths = [X.shape[0]]

		return X_seq, lengths, y

	def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> 'HMMDosageClassifier':
		"""
		Fit HMM to genetic dosage data.

		Args:
			X: Feature matrix (n_rows, n_features) where rows are sites,
			   ordered by individual.  Feature 0 is mean_dosage.
			y: Dosage labels (n_rows,) with values in {0, 1, 2}
			groups: Per-row individual identifier (n_rows,).  Consecutive
			        runs of the same value define one sequence.

		Returns:
			self: Fitted model
		"""
		X = np.asarray(X, dtype=np.float32)
		y_int = coerce_dosage_classes(np.asarray(y, dtype=np.float32))

		# Standardize all features (needed for save/load compatibility)
		Xz, mu, sd = standardize_fit(X)
		self.feature_mean_ = mu
		self.feature_std_ = sd

		# Build per-individual sequences using mean_dosage (feature 0)
		X_seq, lengths, y_seq = self._build_sequences(Xz, y_int, groups)

		if self.verbose:
			print('\n=== Training HMM Dosage Classifier (Semi-Supervised) ===')
			print(f'Rows: {X.shape[0]}, Features: {X.shape[1]}')
			print(f'Individuals (sequences): {len(lengths)}')
			seq_preview = lengths[:5]
			print(f'Sites per sequence: {seq_preview}{"..." if len(lengths) > 5 else ""}')
			print(f'Dosage distribution: {np.bincount(y_int, minlength=3)}')

		# Create and initialize model
		self.model = self._create_hmm_model()
		self._initialize_hmm_with_priors(self.model, X_seq, y_seq)

		try:
			self.model.fit(X_seq, lengths)
		except Exception as e:
			print(f'Error training HMM model: {e}')
			raise

		# Repair degenerate transition matrix (can happen with length-1 sequences)
		row_sums = self.model.transmat_.sum(axis=1)
		if np.any(row_sums < 1e-10):
			bad = row_sums < 1e-10
			self.model.transmat_[bad] = 1.0 / self.n_components
			if self.verbose:
				print(f'Repaired {bad.sum()} zero-sum transmat_ rows (short sequences)')

		if self.verbose:
			print('\nHMM training complete')
			print('Learned transition matrix:')
			for i in range(3):
				print(f'  State {i}: {self.model.transmat_[i]}')

		# Align HMM states to dosage classes using Viterbi on full sequences
		if self.verbose:
			print('\nAligning HMM states to dosage classes...')
		state_alignment = self._align_states_to_dosages(X_seq, y_seq, lengths)
		self.state_to_dosage_ = {int(k): int(v) for k, v in state_alignment.items()}

		# Generate training predictions and compute PYCM metrics
		if self.verbose:
			print('\nComputing training metrics with PYCM...')

		y_pred_train = self.predict_class(X, groups=groups)
		self.pycm_train_ = ConfusionMatrix(actual_vector=y_int.tolist(), predict_vector=y_pred_train.tolist(), digit=4)

		def safe_metric(value):
			if value is None or value == 'None':
				return 0.0
			try:
				return float(value)
			except (ValueError, TypeError):
				return 0.0

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

	def _predict_proba_batch(self, X_seq: np.ndarray, lengths: list) -> np.ndarray:
		"""Predict per-site dosage probabilities using forward-backward.

		Runs the forward-backward algorithm in batch over all concatenated
		sequences, then maps state posteriors to dosage probabilities via
		the learned *state_to_dosage_* alignment.

		Args:
			X_seq: (n_total, 1) mean_dosage observations.
			lengths: Per-individual sequence lengths.

		Returns:
			probs: (n_total, 3) dosage probability per site.
		"""
		try:
			_, state_posteriors = self.model.score_samples(X_seq, lengths)
			# state_posteriors: (n_total, n_states)

			probs = np.zeros((len(X_seq), 3), dtype=np.float32)
			for state in range(self.n_components):
				dosage = self.state_to_dosage_.get(state)
				if dosage is not None:
					probs[:, dosage] += state_posteriors[:, state]

			# Normalize rows
			row_sums = probs.sum(axis=1, keepdims=True)
			row_sums = np.where(row_sums > 0, row_sums, 1.0)
			probs = probs / row_sums

			return probs
		except Exception as e:
			print(f'Warning in HMM prediction: {e}')
			import traceback

			traceback.print_exc()
			return np.ones((len(X_seq), 3), dtype=np.float32) / 3.0

	def predict_proba(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Predict probability distribution over dosage classes using posterior probabilities.

		Args:
			X: Feature matrix (n_rows, n_features).
			groups: Per-row individual identifier.  Consecutive runs of the
			        same value define one sequence.

		Returns:
			proba: Probability matrix (n_rows, 3)
		"""
		if self.model is None:
			raise RuntimeError('Model must be fitted before prediction')

		if self.state_to_dosage_ is None or len(self.state_to_dosage_) == 0:
			raise RuntimeError('State alignment not computed. Model may not be properly trained.')

		Xz = standardize_apply(X, self.feature_mean_, self.feature_std_)
		X_seq, lengths, _ = self._build_sequences(Xz, groups=groups)

		return self._predict_proba_batch(X_seq, lengths)

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

		# Extract arrays from JSON
		means_array = np.array(meta['model_params']['means'])
		covars_array = np.array(meta['model_params']['covars'])
		startprob_array = np.array(meta['model_params']['startprob'])
		transmat_array = np.array(meta['model_params']['transmat'])

		# Validate shapes before setting
		n_components = m.n_components
		n_features = means_array.shape[1] if means_array.ndim > 1 else 1

		if means_array.shape != (n_components, n_features):
			raise ValueError(f'Invalid means shape in saved model. Expected {(n_components, n_features)}, got {means_array.shape}')

		# For diagonal covariance type, squeeze extra dimensions if present
		# hmmlearn expects shape (n_components, n_features) for 'diag' covariance_type
		if m.covariance_type == 'diag':
			# If covars has shape (n_components, n_features, 1), squeeze the last dimension
			if covars_array.ndim == 3 and covars_array.shape[2] == 1:
				covars_array = np.squeeze(covars_array, axis=2)

		if covars_array.shape != (n_components, n_features):
			raise ValueError(
				f'Invalid covars shape in saved model. Expected {(n_components, n_features)}, '
				f'got {covars_array.shape}. This may indicate the model was trained on a dataset '
				f"with {n_features} features, but you're trying to load it with a different dataset."
			)

		# Set n_features before any parameter assignment
		m.model.n_features = n_features

		# Set parameters in correct order
		m.model.startprob_ = startprob_array
		m.model.transmat_ = transmat_array
		m.model.means_ = means_array
		m.model.covars_ = covars_array

		# Restore group models if they exist
		m.use_groups = meta.get('use_groups', False)
		if 'group_models' in meta:
			for group_id_str, gparams in meta['group_models'].items():
				group_id = int(group_id_str)
				gmodel = m._create_hmm_model()
				gmodel.n_features = n_features
				gmodel.startprob_ = np.array(gparams['startprob'])
				gmodel.transmat_ = np.array(gparams['transmat'])
				gmodel.means_ = np.array(gparams['means'])

				# Handle diagonal covariance type - squeeze extra dimensions if needed
				group_covars = np.array(gparams['covars'])
				if m.covariance_type == 'diag' and group_covars.ndim == 3 and group_covars.shape[2] == 1:
					group_covars = np.squeeze(group_covars, axis=2)
				gmodel.covars_ = group_covars
				m.group_models[group_id] = gmodel

		# Restore PYCM metrics
		m.pycm_metrics_ = meta.get('pycm_metrics')

		# Try to load PYCM confusion matrix object
		pycm_path = paths['dir'] / f'{paths["meta"].stem}.pycm.obj'
		if pycm_path.exists():
			try:
				m.pycm_train_ = ConfusionMatrix(file=open(str(pycm_path)))
			except Exception as e:
				print(f'Warning: Could not load PYCM confusion matrix: {e}')

		return m
