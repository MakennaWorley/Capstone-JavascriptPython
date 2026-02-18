from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import arviz as az
import data_preparation
import graph_model_functions
import matplotlib
import numpy as np
import pymc as pm
from sklearn.model_selection import KFold

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# Utilities
# -----------------------------


def ensure_dir(path: str | Path) -> None:
	Path(path).mkdir(parents=True, exist_ok=True)


def flatten_examples(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	X: (n_examples, n_sites, n_features)
	y: (n_examples, n_sites)
	-> (n_examples*n_sites, n_features), (n_examples*n_sites,)
	"""
	if X.ndim != 3:
		raise ValueError(f'Expected X rank-3, got {X.shape}')
	if y.ndim != 2:
		raise ValueError(f'Expected y rank-2, got {y.shape}')
	if X.shape[:2] != y.shape[:2]:
		raise ValueError(f'X/y mismatch: X={X.shape}, y={y.shape}')

	Xf = X.reshape(-1, X.shape[-1]).astype(np.float32, copy=False)
	yf = y.reshape(-1).astype(np.float32, copy=False)
	return Xf, yf


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	mu = X.mean(axis=0)
	sd = X.std(axis=0)
	sd = np.where(sd == 0, 1.0, sd)
	return (X - mu) / sd, mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
	return (X - mu) / sd


def coerce_dosage_classes(y: np.ndarray) -> np.ndarray:
	"""
	y can be float-ish dosage; this coerces to int {0,1,2}.
	"""
	y_int = np.rint(y).astype(np.int64)
	y_int = np.clip(y_int, 0, 2)
	return y_int


# -----------------------------
# Persistence helpers
# -----------------------------


def model_paths(models_dir: str | Path, base_name: str, model_tag: str) -> Dict[str, Path]:
	d = Path(models_dir)
	return {
		'dir': d,
		'idata': d / f'{base_name}.{model_tag}.idata.nc',
		'meta': d / f'{base_name}.{model_tag}.meta.json',
		'graph_val': d / f'{base_name}.{model_tag}.val_plot.png',
		'graph_test': d / f'{base_name}.{model_tag}.test_plot.png',
	}


def save_common_meta(paths: Dict[str, Path], payload: Dict[str, Any]) -> None:
	ensure_dir(paths['dir'])
	paths['meta'].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def load_meta(paths: Dict[str, Path]) -> Dict[str, Any]:
	return json.loads(paths['meta'].read_text(encoding='utf-8'))


# -----------------------------
# Model 1: Bayesian Linear Dosage Regression
# -----------------------------


class BayesianLinearDosageRegressor:
	def __init__(
		self,
		*,
		draws: int = 1000,
		tune: int = 1000,
		chains: int = 4,
		target_accept: float = 0.9,
		random_seed: int = 123,
		clip_to_dosage_range: bool = True,
	):
		self.draws = draws
		self.tune = tune
		self.chains = chains
		self.target_accept = target_accept
		self.random_seed = random_seed
		self.clip_to_dosage_range = clip_to_dosage_range

		self.idata: Optional[az.InferenceData] = None
		self.feature_mean_: Optional[np.ndarray] = None
		self.feature_std_: Optional[np.ndarray] = None

		self._coef_mean: Optional[np.ndarray] = None
		self._intercept_mean: Optional[float] = None

	@property
	def tag(self) -> str:
		return 'bayes_linear'

	def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> 'BayesianLinearDosageRegressor':
		X = np.asarray(X, dtype=np.float32)
		y = np.asarray(y, dtype=np.float32)

		Xz, mu, sd = standardize_fit(X)

		# Guard
		Xz = np.clip(Xz, -10.0, 10.0)

		self.feature_mean_ = mu
		self.feature_std_ = sd
		n_groups = len(np.unique(groups))

		n, k = Xz.shape

		with pm.Model() as model:
			# Hierarchical Priors (Population Level)
			mu_alpha = pm.Normal('mu_alpha', mu=0.0, sigma=1.0)
			sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1.0)

			# Group-specific intercepts (Ancestry branches)
			intercepts = pm.Normal('intercepts', mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)
			coef = pm.Normal('coef', mu=0.0, sigma=1.0, shape=Xz.shape[1])
			sigma = pm.HalfNormal('sigma', sigma=1.0)

			mu_y = intercepts[groups] + pm.math.dot(Xz, coef)
			pm.Normal('y', mu=mu_y, sigma=sigma, observed=y)

			self.idata = pm.sample(
				draws=self.draws,
				tune=self.tune,
				chains=self.chains,
				target_accept=self.target_accept,
				random_seed=self.random_seed,
				return_inferencedata=True,
			)

		self._coef_mean = self.idata.posterior['coef'].mean(axis=(0, 1)).values
		self._mu_alpha_mean = float(self.idata.posterior['mu_alpha'].mean())
		return self

	def predict(self, X: np.ndarray) -> np.ndarray:
		Xz = standardize_apply(X, self.feature_mean_, self.feature_std_)
		yhat = self._mu_alpha_mean + Xz @ self._coef_mean
		return np.clip(yhat, 0.0, 2.0) if self.clip_to_dosage_range else yhat

	def get_calibration_data(self):
		if self.idata is None:
			raise RuntimeError('Model must be fit first.')
		with pm.Model():  # Re-create model context for PPC
			ppc = pm.sample_posterior_predictive(self.idata)
		return ppc

	def save(self, paths: Dict[str, Path], extra_meta: Dict[str, Any]) -> None:
		if self.idata is None:
			raise RuntimeError('No idata to save.')
		ensure_dir(paths['dir'])
		az.to_netcdf(self.idata, paths['idata'])

		payload = {
			'type': 'BayesianLinearDosageRegressor',
			'feature_mean': self.feature_mean_.tolist(),
			'feature_std': self.feature_std_.tolist(),
			'posterior_means': {'coef': self._coef_mean.tolist(), 'mu_alpha': self._mu_alpha_mean},
			'params': {
				'draws': self.draws,
				'tune': self.tune,
				'chains': self.chains,
				'target_accept': self.target_accept,
				'random_seed': self.random_seed,
				'clip_to_dosage_range': self.clip_to_dosage_range,
			},
			'extra': extra_meta,
		}
		save_common_meta(paths, payload)

	@classmethod
	def load(cls, paths: Dict[str, Path]) -> 'BayesianLinearDosageRegressor':
		meta = load_meta(paths)
		m = cls(**meta['params'])
		m.idata = az.from_netcdf(paths['idata'])

		m.feature_mean_ = np.array(meta['feature_mean'], dtype=np.float32)
		m.feature_std_ = np.array(meta['feature_std'], dtype=np.float32)
		m._coef_mean = np.array(meta['posterior_means']['coef'], dtype=np.float32)
		m._mu_alpha_mean = float(meta['posterior_means']['mu_alpha'])
		return m


# -----------------------------
# Model 2: Bayesian Categorical Dosage (Softmax)
# -----------------------------


class BayesianCategoricalDosageClassifier:
	"""
	Multinomial logistic regression with softmax over 3 classes (0/1/2).
	We return:
	  - predict_proba(X): (n, 3)
	  - predict_class(X): (n,)
	  - predict(X): expected dosage E[y] for compatibility with regression plotting
	"""

	def __init__(self, *, draws: int = 1000, tune: int = 1000, chains: int = 2, target_accept: float = 0.9, random_seed: int = 123):
		self.draws = draws
		self.tune = tune
		self.chains = chains
		self.target_accept = target_accept
		self.random_seed = random_seed

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
		y_int = coerce_dosage_classes(np.asarray(y, dtype=np.float32))

		Xz, mu, sd = standardize_fit(X)

		self.feature_mean_ = mu
		self.feature_std_ = sd
		n_groups = len(np.unique(groups))
		C = 3

		with pm.Model() as model:
			# Hierarchical Priors for the Intercepts
			mu_b = pm.Normal('mu_b', mu=0.0, sigma=1.0, shape=C)
			sigma_b = pm.HalfNormal('sigma_b', sigma=1.0, shape=C)

			# Group-level intercepts
			b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=(n_groups, C))
			W = pm.Normal('W', mu=0.0, sigma=1.0, shape=(Xz.shape[1], C))

			logits = b[groups] + pm.math.dot(Xz, W)
			pm.Categorical('y', logit_p=logits, observed=y_int)

			self.idata = pm.sample(
				draws=self.draws,
				tune=self.tune,
				chains=self.chains,
				target_accept=self.target_accept,
				random_seed=self.random_seed,
				return_inferencedata=True,
			)

		self._W_mean = self.idata.posterior['W'].mean(axis=(0, 1)).values
		self._mu_b_mean = self.idata.posterior['mu_b'].mean(axis=(0, 1)).values
		return self

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		Xz = standardize_apply(X, self.feature_mean_, self.feature_std_)
		logits = self._mu_b_mean + Xz @ self._W_mean
		expz = np.exp(logits - logits.max(axis=1, keepdims=True))
		return (expz / expz.sum(axis=1, keepdims=True)).astype(np.float32)

	def get_calibration_data(self):
		if self.idata is None:
			raise RuntimeError('Model must be fit first.')
		with pm.Model():
			ppc = pm.sample_posterior_predictive(self.idata)
		return ppc

	def predict_class(self, X: np.ndarray) -> np.ndarray:
		p = self.predict_proba(X)
		return np.argmax(p, axis=1).astype(np.int64)

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""
		Expected dosage E[y] = sum_c c * p(c)
		This makes it compatible with regression metrics/plots.
		"""
		p = self.predict_proba(X)
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
		save_common_meta(paths, payload)

	@classmethod
	def load(cls, paths: Dict[str, Path]) -> 'BayesianCategoricalDosageClassifier':
		meta = load_meta(paths)
		m = cls(**meta['params'])
		m.idata = az.from_netcdf(paths['idata'])

		m.feature_mean_ = np.array(meta['feature_mean'], dtype=np.float32)
		m.feature_std_ = np.array(meta['feature_std'], dtype=np.float32)
		m._W_mean = np.array(meta['posterior_means']['W'], dtype=np.float32)
		m._mu_b_mean = np.array(meta['posterior_means']['mu_b'], dtype=np.float32)
		return m


# -----------------------------
# Pipeline
# -----------------------------


class _NoRefitProxy:
	"""
	graph_model_functions.evaluate_and_graph_reg calls .fit().
	We wrap a fitted model so fit() is a no-op.
	"""

	def __init__(self, fitted_model):
		self._m = fitted_model

	def fit(self, X, y):
		return self

	def predict(self, X):
		return self._m.predict(X)


def _prep_triplet(
	base_name: str, prep_cfg: data_preparation.PrepConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	triplet = data_preparation.prepare_data_triplet(base_name, prep_cfg)

	X_train, y_train = flatten_examples(triplet['train']['X'], triplet['train']['y'])
	X_val, y_val = flatten_examples(triplet['val']['X'], triplet['val']['y'])
	X_test, y_test = flatten_examples(triplet['test']['X'], triplet['test']['y'])

	# Extract and flatten the hierarchy/group IDs
	groups = {}
	for split, X_set in [('train', X_train), ('val', X_val), ('test', X_test)]:
		if 'groups' in triplet[split]:
			groups[split] = triplet[split]['groups'].flatten().astype(int)
		else:
			# Fallback: Assign everyone to Group 0 (Non-hierarchical mode)
			print(f"Warning: 'groups' key missing in {split} split. Using default Group 0.")
			groups[split] = np.zeros(X_set.shape[0], dtype=int)

	return X_train, y_train, X_val, y_val, X_test, y_test, groups


def train_with_cross_val(base_name, model_kind, prep_cfg, n_splits=5, models_dir='models'):
	"""
	Performs K-Fold CV on the combined training and validation sets
	and returns a model fit on the full combined dataset.
	"""
	X_train, y_train, X_val, y_val, X_test, y_test, groups = _prep_triplet(base_name, prep_cfg)

	# Combine train and val: the model updates on all available non-test data
	X_all = np.concatenate([X_train, X_val], axis=0)
	y_all = np.concatenate([y_train, y_val], axis=0)
	groups_all = np.concatenate([groups['train'], groups['val']], axis=0)

	kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

	print(f'\n--- Starting {n_splits}-Fold Cross-Validation for {model_kind} ---')

	for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
		X_t, X_v = X_all[train_idx], X_all[val_idx]
		y_t, y_v = y_all[train_idx], y_all[val_idx]
		g_t = groups_all[train_idx]

		# Temporary model for fold validation
		if model_kind == 'linear':
			fold_model = BayesianLinearDosageRegressor()
		else:
			fold_model = BayesianCategoricalDosageClassifier()

		fold_model.fit(X_t, y_t, groups=g_t)

		# Calculate metrics for monitoring
		y_pred = fold_model.predict(X_v)
		mse = np.mean((y_v - y_pred) ** 2)
		print(f'Fold {fold + 1} MSE: {mse:.4f}')

	# Final Step: Return a model fit on the ENTIRE combined pool
	print(f'--- Fitting final {model_kind} model on all CV data ---')
	if model_kind == 'linear':
		final_model = BayesianLinearDosageRegressor()
	else:
		final_model = BayesianCategoricalDosageClassifier()

	final_model.fit(X_all, y_all, groups=groups_all)
	return final_model


def train_eval_one(
	base_name: str,
	model_kind: str,
	*,
	prep_cfg: Optional[data_preparation.PrepConfig] = None,
	models_dir: str | Path = 'models',
	force_retrain: bool = False,
	graph: bool = True,
	draws: int = 1000,
	tune: int = 1000,
	chains: int = 2,
	target_accept: float = 0.9,
	seed: int = 123,
) -> Dict[str, Any]:
	if prep_cfg is None:
		prep_cfg = data_preparation.PrepConfig(dataset_name='unused')

	X_train, y_train, X_val, y_val, X_test, y_test, groups = _prep_triplet(base_name, prep_cfg)

	if model_kind == 'linear':
		ModelCls = BayesianLinearDosageRegressor
		model_tag = BayesianLinearDosageRegressor().tag
	elif model_kind == 'softmax3':
		ModelCls = BayesianCategoricalDosageClassifier
		model_tag = BayesianCategoricalDosageClassifier().tag
	else:
		raise ValueError("model_kind must be 'linear' or 'softmax3'")

	paths = model_paths(models_dir, base_name, model_tag)

	if (not force_retrain) and paths['meta'].exists() and paths['idata'].exists():
		model = ModelCls.load(paths)
		trained = False
	else:
		# Check if we are dealing with validation data to trigger cross-validation
		if 'validation' in base_name.lower():
			print(f"Validation detected in '{base_name}'. Updating model using cross-validation...")
			model = train_with_cross_val(base_name, model_kind, prep_cfg, models_dir=models_dir)
		else:
			model = ModelCls(draws=draws, tune=tune, chains=chains, target_accept=target_accept, random_seed=seed)
			model.fit(X_train, y_train, groups=groups['train'])

		trained = True
		model.get_calibration_data()
		model.save(paths, extra_meta={'base_name': base_name, 'prep_cfg': asdict(prep_cfg)})

	proxy = _NoRefitProxy(model)

	# --- Validation Step ---
	print(f'\n=== {model_tag.upper()} (Validation) ===')
	val_metrics = graph_model_functions.evaluate_and_graph_reg(
		proxy, X_train, y_train, X_val, y_val, name=f'{model_tag}[{base_name}] (val)', graph=True
	)

	# --- Validation Graph ---
	if paths['graph_val']:
		plt.savefig(paths['graph_val'])
		print(f'Graph saved to: {paths["graph_val"]}')
	plt.close()

	# --- Test Step ---
	print(f'\n=== {model_tag.upper()} (Test) ===')
	test_metrics = graph_model_functions.evaluate_and_graph_reg(
		proxy, X_train, y_train, X_test, y_test, name=f'{model_tag}[{base_name}] (test)', graph=True
	)

	# --- Test Graph ---
	plt.savefig(paths['graph_test'])
	plt.close()

	extra = {}
	if model_kind == 'softmax3':
		yv = coerce_dosage_classes(y_val)
		yt = coerce_dosage_classes(y_test)
		val_acc = (model.predict_class(X_val) == yv).mean()
		test_acc = (model.predict_class(X_test) == yt).mean()
		extra['val_accuracy'] = float(val_acc)
		extra['test_accuracy'] = float(test_acc)
		print(f'\n[softmax3] val accuracy:  {val_acc:.4f}')
		print(f'[softmax3] test accuracy: {test_acc:.4f}')

	return {
		'model_kind': model_kind,
		'trained': trained,
		'paths': {k: str(v) for k, v in paths.items() if k != 'dir'},
		'val': val_metrics,
		'test': test_metrics,
		**extra,
	}


def train_eval_both(
	base_name: str,
	*,
	prep_cfg: Optional[data_preparation.PrepConfig] = None,
	models_dir: str | Path = 'models',
	force_retrain: bool = False,
	graph: bool = True,
	draws: int = 1000,
	tune: int = 1000,
	chains: int = 2,
	target_accept: float = 0.9,
	seed: int = 123,
) -> Dict[str, Any]:
	out_linear = train_eval_one(
		base_name,
		'linear',
		prep_cfg=prep_cfg,
		models_dir=models_dir,
		force_retrain=force_retrain,
		graph=graph,
		draws=draws,
		tune=tune,
		chains=chains,
		target_accept=target_accept,
		seed=seed,
	)

	out_softmax = train_eval_one(
		base_name,
		'softmax3',
		prep_cfg=prep_cfg,
		models_dir=models_dir,
		force_retrain=force_retrain,
		graph=graph,
		draws=draws,
		tune=tune,
		chains=chains,
		target_accept=target_accept,
		seed=seed,
	)

	return {'linear': out_linear, 'softmax3': out_softmax}


def test_on_new_data(model, dataset_name: str, prep_cfg: Optional[data_preparation.PrepConfig] = None):
	"""
	Loads a new dataset, prepares it using the model's original scaling params,
	and returns predictions and metrics without any training.
	"""
	if prep_cfg is None:
		prep_cfg = data_preparation.PrepConfig(dataset_name=dataset_name)

	# 1. Load the new data
	new_data = data_preparation.prepare_data(dataset_name, prep_cfg)
	X_raw, y_raw = flatten_examples(new_data['test']['X'], new_data['test']['y'])

	# 2. Wrap the model in the NoRefitProxy to ensure .fit() cannot be called
	proxy = _NoRefitProxy(model)

	print(f'\n=== Testing on New Dataset: {dataset_name} ===')

	# 3. Use the existing evaluation utility to get metrics and plots
	# This uses model.predict() internally, which applies stored feature_mean_ and feature_std_
	metrics = graph_model_functions.evaluate_and_graph_reg(proxy, None, None, X_raw, y_raw, name=f'External_Test_{dataset_name}', graph=True)

	return metrics


if __name__ == '__main__':
	results = train_eval_both(
		base_name='testing', models_dir='models', force_retrain=False, graph=False, draws=1000, tune=1000, chains=4, target_accept=0.9, seed=123
	)
	print('\nDone:', results)
