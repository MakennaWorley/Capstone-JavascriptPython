import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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

	def predict_proba(self, X: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
		X_scaled = self.scaler.transform(X)
		return self.model.predict_proba(X_scaled).astype(np.float32)

	def predict_class(self, X: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
		X_scaled = self.scaler.transform(X)
		return self.model.predict(X_scaled)

	def predict(self, X: np.ndarray, groups: np.ndarray = None) -> np.ndarray:
		"""
		Returns expected dosage E[y] for regression metric compatibility.
		"""
		p = self.predict_proba(X)
		classes = np.array([0.0, 1.0, 2.0])
		return (p * classes[None, :]).sum(axis=1)

	def save(self, paths: Dict[str, Path], extra_meta: Dict[str, Any]) -> None:
		"""
		Saves the model parameters to JSON, matching the Bayesian model's style.
		"""
		# We extract the raw coefficients and intercepts from sklearn
		payload = {
			'type': 'SklearnMultinomialClassifier',
			'tag': self.tag,
			'feature_mean': self.scaler.mean_.tolist(),
			'feature_std': self.scaler.scale_.tolist(),
			'coef': self.model.coef_.tolist(),  # The 'W' equivalent
			'intercept': self.model.intercept_.tolist(),  # The 'b' equivalent
			'params': self.model.get_params(),
			'extra': extra_meta,
		}

		# Using your utility function from model_bayesian.py
		paths['meta'].write_text(json.dumps(payload, indent=2), encoding='utf-8')

	@classmethod
	def load(cls, paths: Dict[str, Path]) -> 'SklearnMultinomialClassifier':
		meta = json.loads(paths['meta'].read_text(encoding='utf-8'))
		m = cls()

		# Reconstruct the scaler
		m.scaler.mean_ = np.array(meta['feature_mean'])
		m.scaler.scale_ = np.array(meta['feature_std'])

		# Reconstruct the sklearn model without needing to re-fit
		m.model.coef_ = np.array(meta['coef'])
		m.model.intercept_ = np.array(meta['intercept'])
		m.model.classes_ = np.array([0, 1, 2])  # Manually set for dosage
		return m
