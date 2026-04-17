from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pymc as pm

from .model_functions import coerce_dosage_classes, ensure_dir, load_meta, save_common_meta, standardize_apply, standardize_fit

# Configure PyMC to use JAX backend for GPU acceleration (if available)
try:
	import os

	# Configure JAX for multiprocessing compatibility BEFORE importing
	os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
	os.environ.setdefault('JAX_ENABLE_X64', 'false')

	import jax
	import jax.numpy as jnp

	# Configure JAX for optimal performance
	jax.config.update('jax_enable_x64', False)  # Use float32 for speed

	# Memory optimization
	os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.8')  # Use 80% of GPU memory
	os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

	# Check if CUDA is available
	if jax.devices('gpu'):
		print(f'GPU devices found: {jax.devices("gpu")}')
		# Enable GPU for computations
		jax.config.update('jax_platform_name', 'gpu')
		# Enable memory preallocation for better performance
		os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
	else:
		print('No GPU devices found, using CPU')
		jax.config.update('jax_platform_name', 'cpu')
except ImportError:
	print('JAX not installed, using default PyMC backend (CPU)')
except Exception as e:
	print(f'GPU setup warning: {e}')


class BayesianCategoricalDosageClassifier:
	"""
	Multinomial logistic regression with softmax over 3 classes (0/1/2).
	Returns:
	  - predict_proba(X): (n, 3)
	  - predict_class(X): (n,)
	  - predict(X): expected dosage E[y] for compatibility with regression plotting
	"""

	def __init__(
		self,
		*,
		draws: int = 1000,
		tune: int = 1000,
		chains: int = 4,
		target_accept: float = 0.95,
		random_seed: Optional[int] = None,
		cores: int = 8,
		use_gpu: bool = True,
		gpu_strategy: str = 'aggressive',  # 'safe' uses 4 cores, 'aggressive' uses all cores
	):
		self.draws = draws
		self.tune = tune
		self.chains = chains
		self.target_accept = target_accept
		if random_seed is None:
			random_seed = int(np.random.SeedSequence().entropy % (2**32))
		self.random_seed = random_seed
		self.cores = cores
		self.use_gpu = use_gpu
		self.gpu_strategy = gpu_strategy

		# Check GPU availability
		self.gpu_available = False
		try:
			import jax

			self.gpu_available = len(jax.devices('gpu')) > 0 and self.use_gpu
			if self.gpu_available:
				print(f'GPU acceleration enabled with {len(jax.devices("gpu"))} GPU(s)')
		except:
			pass

		self.idata: Optional[az.InferenceData] = None
		self.feature_mean_: Optional[np.ndarray] = None
		self.feature_std_: Optional[np.ndarray] = None

		# posterior means for fast inference
		self._w_mean: Optional[np.ndarray] = None  # (k, 3)
		self._b_mean: Optional[np.ndarray] = None  # (3,)

	@property
	def tag(self) -> str:
		return 'bayes_softmax3'

	def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> 'BayesianCategoricalDosageClassifier':
		X = np.asarray(X, dtype=np.float32)
		y_int = coerce_dosage_classes(np.asarray(y, dtype=np.float32))

		Xz, mu, sd = standardize_fit(X)

		self.feature_mean_ = mu
		self.feature_std_ = sd
		n_groups = len(np.unique(groups))
		C = 3

		with pm.Model():
			# 1. Define Hierarchical Priors for category intercepts
			mu_b = pm.Normal('mu_b', mu=0.0, sigma=1.0, shape=C)
			sigma_b = pm.HalfNormal('sigma_b', sigma=1.0, shape=C)

			# 2. Define Group-level intercepts per category
			b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=(n_groups, C))
			W = pm.Normal('W', mu=0.0, sigma=5.0, shape=(Xz.shape[1], C))

			# 3. Logits calculation using the group indices
			logits = b[groups] + pm.math.dot(Xz, W)
			pm.Categorical('y', logit_p=logits, observed=y_int)

			if self.gpu_available:
				if self.gpu_strategy == 'safe':
					# Balanced GPU Mode: Parallel chains + GPU, but fewer chains for stability
					effective_chains = min(self.chains, 4)  # Keep your original chain count
					effective_cores = min(self.cores, 4)  # Use multiple cores but not all
					print(f'GPU Balanced Mode: {effective_chains} chains across {effective_cores} cores')
					print('  Parallel chains + GPU acceleration (fork warnings are OK)')
				else:
					# Aggressive GPU Mode: Use everything
					effective_chains = self.chains
					effective_cores = self.cores
					print(f'GPU Max Mode: {effective_chains} chains across {effective_cores} cores')
					print('  Maximum parallelism + GPU (ignore fork warnings)')
			else:
				# CPU Mode: Use all resources
				effective_chains = self.chains
				effective_cores = self.cores
				print(f'CPU Mode: {effective_chains} chains across {effective_cores} cores')

			self.idata = pm.sample(
				draws=self.draws,
				tune=self.tune,
				chains=effective_chains,
				target_accept=self.target_accept,
				random_seed=self.random_seed,
				return_inferencedata=True,
				cores=effective_cores,
			)

		self._w_mean = self.idata.posterior['W'].mean(axis=(0, 1)).values
		self._mu_b_mean = self.idata.posterior['mu_b'].mean(axis=(0, 1)).values
		return self

	def predict_proba(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		if self.idata is None:
			raise RuntimeError('Model must be fitted before prediction')

		Xz = standardize_apply(X, self.feature_mean_, self.feature_std_)

		if groups is not None:
			# b shape is (n_groups, 3)
			group_b = self.idata.posterior['b'].mean(axis=(0, 1)).values
			max_group = len(group_b) - 1

			# Warn about invalid groups
			if np.any((groups > max_group) | (groups < 0)):
				n_invalid = np.sum((groups > max_group) | (groups < 0))
				print(f'Warning: {n_invalid} samples have invalid group indices (valid range: 0-{max_group}). Using global mean.')

			intercept = np.take(group_b, groups, axis=0, mode='clip')
			mask = (groups > max_group) | (groups < 0)
			intercept[mask] = self._mu_b_mean
		else:
			# Fallback to global category means
			intercept = self._mu_b_mean  # shape (3,)

		logits = intercept + Xz @ self._w_mean
		expz = np.exp(logits - logits.max(axis=1, keepdims=True))
		return (expz / expz.sum(axis=1, keepdims=True)).astype(np.float32)

	def predict_class(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		p = self.predict_proba(X, groups=groups)
		return np.argmax(p, axis=1).astype(np.int64)

	def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Expected dosage E[y] = sum_c c * p(c)
		This makes it compatible with regression metrics/plots.
		"""
		p = self.predict_proba(X, groups=groups)
		classes = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		return (p * classes[None, :]).sum(axis=1)

	def save(self, paths: Dict[str, Path], extra_meta: Dict[str, Any]) -> None:
		if self.idata is None:
			raise RuntimeError('No idata to save.')
		ensure_dir(paths['dir'])
		az.to_netcdf(self.idata, paths['idata'])

		payload = {
			'type': 'BayesianCategoricalDosageClassifier',
			'tag': self.tag,
			'feature_mean': self.feature_mean_.tolist(),
			'feature_std': self.feature_std_.tolist(),
			'posterior_means': {'W': self._w_mean.tolist(), 'mu_b': self._mu_b_mean.tolist()},
			'params': {
				'draws': self.draws,
				'tune': self.tune,
				'chains': self.chains,
				'target_accept': self.target_accept,
				'random_seed': self.random_seed,
				'cores': self.cores,
				'use_gpu': self.use_gpu,
				'gpu_strategy': self.gpu_strategy,
			},
			'extra': extra_meta,
		}
		save_common_meta(paths, payload)

	@classmethod
	def load(cls, paths: Dict[str, Path]) -> 'BayesianCategoricalDosageClassifier':
		meta = load_meta(paths)
		m = cls(**meta['params'])
		m.idata = az.from_netcdf(paths['idata'])

		m.feature_mean_ = np.array(meta['feature_mean'], dtype=np.float32)
		m.feature_std_ = np.array(meta['feature_std'], dtype=np.float32)
		m._w_mean = np.array(meta['posterior_means']['W'], dtype=np.float32)
		m._mu_b_mean = np.array(meta['posterior_means']['mu_b'], dtype=np.float32)
		return m
