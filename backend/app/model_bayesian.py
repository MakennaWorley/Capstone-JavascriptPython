from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import arviz as az
import numpy as np
import pymc as pm
from model_functions import _coerce_dosage_classes, _ensure_dir, _load_meta, _save_common_meta, _standardize_apply, _standardize_fit


class BayesianCategoricalDosageClassifier:
	"""
	Multinomial logistic regression with softmax over 3 classes (0/1/2).
	Returns:
	  - predict_proba(X): (n, 3)
	  - predict_class(X): (n,)
	  - predict(X): expected dosage E[y] for compatibility with regression plotting
	"""

	def __init__(self, *, draws: int = 1000, tune: int = 1000, chains: int = 4, target_accept: float = 0.9, random_seed: int = 123, cores: int = 8):
		self.draws = draws
		self.tune = tune
		self.chains = chains
		self.target_accept = target_accept
		self.random_seed = random_seed
		self.cores = cores

		self.idata: Optional[az.InferenceData] = None
		self.feature_mean_: Optional[np.ndarray] = None
		self.feature_std_: Optional[np.ndarray] = None

		# posterior means for fast inference
		self._W_mean: Optional[np.ndarray] = None  # (k, 3)
		self._b_mean: Optional[np.ndarray] = None  # (3,)

	@property
	def tag(self) -> str:
		return 'bayes_softmax3'

	def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> 'BayesianCategoricalDosageClassifier':
		X = np.asarray(X, dtype=np.float32)
		y_int = _coerce_dosage_classes(np.asarray(y, dtype=np.float32))

		Xz, mu, sd = _standardize_fit(X)

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
			W = pm.Normal('W', mu=0.0, sigma=1.0, shape=(Xz.shape[1], C))

			# 3. Logits calculation using the group indices
			logits = b[groups] + pm.math.dot(Xz, W)
			pm.Categorical('y', logit_p=logits, observed=y_int)

			self.idata = pm.sample(
				draws=self.draws,
				tune=self.tune,
				chains=self.chains,
				target_accept=self.target_accept,
				random_seed=self.random_seed,
				return_inferencedata=True,
				cores=self.cores,
			)

		self._W_mean = self.idata.posterior['W'].mean(axis=(0, 1)).values
		self._mu_b_mean = self.idata.posterior['mu_b'].mean(axis=(0, 1)).values
		return self

	def predict_proba(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		Xz = _standardize_apply(X, self.feature_mean_, self.feature_std_)

		if self.idata is not None and groups is not None:
			# b shape is (n_groups, 3)
			group_b = self.idata.posterior['b'].mean(axis=(0, 1)).values
			intercept = np.take(group_b, groups, axis=0, mode='clip')
			mask = (groups >= len(group_b)) | (groups < 0)
			intercept[mask] = self._mu_b_mean
		else:
			# Fallback to global category means
			intercept = self._mu_b_mean  # shape (3,)

		logits = intercept + Xz @ self._W_mean
		expz = np.exp(logits - logits.max(axis=1, keepdims=True))
		return (expz / expz.sum(axis=1, keepdims=True)).astype(np.float32)

	def get_calibration_data(self):
		if self.idata is None:
			raise RuntimeError('Model must be fit first.')
		with pm.Model():
			ppc = pm.sample_posterior_predictive(self.idata)
		return ppc

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
		_ensure_dir(paths['dir'])
		az.to_netcdf(self.idata, paths['idata'])

		payload = {
			'type': 'BayesianCategoricalDosageClassifier',
			'tag': self.tag,
			'feature_mean': self.feature_mean_.tolist(),
			'feature_std': self.feature_std_.tolist(),
			'posterior_means': {'W': self._W_mean.tolist(), 'mu_b': self._mu_b_mean.tolist()},
			'params': {
				'draws': self.draws,
				'tune': self.tune,
				'chains': self.chains,
				'target_accept': self.target_accept,
				'random_seed': self.random_seed,
			},
			'extra': extra_meta,
		}
		_save_common_meta(paths, payload)

	@classmethod
	def load(cls, paths: Dict[str, Path]) -> 'BayesianCategoricalDosageClassifier':
		meta = _load_meta(paths)
		m = cls(**meta['params'])
		m.idata = az.from_netcdf(paths['idata'])

		m.feature_mean_ = np.array(meta['feature_mean'], dtype=np.float32)
		m.feature_std_ = np.array(meta['feature_std'], dtype=np.float32)
		m._W_mean = np.array(meta['posterior_means']['W'], dtype=np.float32)
		m._mu_b_mean = np.array(meta['posterior_means']['mu_b'], dtype=np.float32)
		return m
