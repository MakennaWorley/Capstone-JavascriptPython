from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
import numpy as np

matplotlib.use('Agg')


class _NoRefitProxy:
	"""
	graph_model_functions.evaluate_and_graph_clf calls .fit().
	We wrap a fitted model so fit() is a no-op.
	"""

	def __init__(self, fitted_model):
		self._m = fitted_model

	def fit(self, X, y):
		return self

	def predict(self, X):
		return self._m.predict(X)


def _ensure_dir(path: str | Path) -> None:
	Path(path).mkdir(parents=True, exist_ok=True)


def _flatten_examples(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	mu = X.mean(axis=0)
	sd = X.std(axis=0)
	sd = np.where(sd == 0, 1.0, sd)
	return (X - mu) / sd, mu, sd


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
	return (X - mu) / sd


def _coerce_dosage_classes(y: np.ndarray) -> np.ndarray:
	"""
	y can be float-ish dosage; this coerces to int {0,1,2}.
	"""
	y_int = np.rint(y).astype(np.int64)
	y_int = np.clip(y_int, 0, 2)
	return y_int


def _model_paths(models_dir: str | Path, base_name: str, model_tag: str) -> Dict[str, Path]:
	d = Path(models_dir)
	return {
		'dir': d,
		'idata': d / f'{base_name}.{model_tag}.idata.nc',
		'meta': d / f'{base_name}.{model_tag}.meta.json',
		'graph_test': d / f'{base_name}.{model_tag}.test_plot.png',
		'graph_cm': d / f'{base_name}.{model_tag}.cm_plot.png',
	}


def _save_common_meta(paths: Dict[str, Path], payload: Dict[str, Any]) -> None:
	_ensure_dir(paths['dir'])
	paths['meta'].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def _load_meta(paths: Dict[str, Path]) -> Dict[str, Any]:
	return json.loads(paths['meta'].read_text(encoding='utf-8'))
