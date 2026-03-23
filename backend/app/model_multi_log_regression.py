import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def ensure_dir(p: Path) -> None:
	"""Ensure directory exists."""
	p.mkdir(parents=True, exist_ok=True)


class SklearnMultinomialClassifier:
	"""
	A standard Multinomial Logistic Regression baseline using scikit-learn.
	This provides a 'simple' Frequentist comparison to your Bayesian model.
	"""

	def __init__(self, random_seed=123):
		self.model = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000, random_state=random_seed)
		self.scaler = StandardScaler()

	@property
	def tag(self) -> str:
		return 'multi_log_regression'

	def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray = None):
		"""
		Fits the model using Maximum Likelihood Estimation (MLE).
		Note: Standard Logistic Regression doesn't natively use the 'groups'
		parameter like your hierarchical Bayesian model does.
		"""
		# 1. Coerce dosage to discrete integers {0, 1, 2}
		y_int = np.rint(y).astype(int)

		# 2. Standardize features
		X_scaled = self.scaler.fit_transform(X)

		# 3. Fit the model
		self.model.fit(X_scaled, y_int)
		print(f'[{self.tag}] Model fitted successfully.')
		return self

	def predict_proba(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		if not hasattr(self.scaler, 'mean_'):
			raise RuntimeError('Model must be fitted before prediction')
		if not hasattr(self.model, 'coef_'):
			raise RuntimeError('Model must be fitted before prediction')
		X_scaled = self.scaler.transform(X)
		return self.model.predict_proba(X_scaled).astype(np.float32)

	def predict_class(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		if not hasattr(self.scaler, 'mean_'):
			raise RuntimeError('Model must be fitted before prediction')
		X_scaled = self.scaler.transform(X)
		return self.model.predict(X_scaled)

	def predict(self, X: np.ndarray, groups: Optional[np.ndarray] = None) -> np.ndarray:
		"""
		Returns expected dosage E[y] for regression metric compatibility.
		"""
		p = self.predict_proba(X, groups=groups)
		classes = np.array([0.0, 1.0, 2.0])
		return (p * classes[None, :]).sum(axis=1)

	def save(self, paths: Dict[str, Path], extra_meta: Dict[str, Any]) -> None:
		"""
		Saves the model parameters to JSON, matching the Bayesian model's style.
		"""
		if not hasattr(self.model, 'coef_'):
			raise RuntimeError('Model must be fitted before saving')

		ensure_dir(paths['dir'])

		# We extract the raw coefficients and intercepts from sklearn
		payload = {
			'type': 'SklearnMultinomialClassifier',
			'tag': self.tag,
			'feature_mean': self.scaler.mean_.tolist(),
			'feature_std': self.scaler.scale_.tolist(),
			'n_features': int(self.scaler.n_features_in_),
			'coef': self.model.coef_.tolist(),  # The 'W' equivalent
			'intercept': self.model.intercept_.tolist(),  # The 'b' equivalent
			'params': self.model.get_params(),
			'extra': extra_meta,
		}

		paths['meta'].write_text(json.dumps(payload, indent=2), encoding='utf-8')

	@classmethod
	def load(cls, paths: Dict[str, Path]) -> 'SklearnMultinomialClassifier':
		meta = json.loads(paths['meta'].read_text(encoding='utf-8'))
		m = cls()

		# Reconstruct the scaler
		m.scaler.mean_ = np.array(meta['feature_mean'])
		m.scaler.scale_ = np.array(meta['feature_std'])
		m.scaler.n_features_in_ = meta.get('n_features', len(meta['feature_mean']))
		m.scaler.var_ = m.scaler.scale_**2  # Required by sklearn

		# Reconstruct the sklearn model without needing to re-fit
		m.model.coef_ = np.array(meta['coef'])
		m.model.intercept_ = np.array(meta['intercept'])
		m.model.classes_ = np.array([0, 1, 2])  # Manually set for dosage
		m.model.n_features_in_ = meta.get('n_features', len(meta['feature_mean']))
		m.model.n_iter_ = np.array([meta['params'].get('max_iter', 1000)])  # Required by sklearn
		return m
