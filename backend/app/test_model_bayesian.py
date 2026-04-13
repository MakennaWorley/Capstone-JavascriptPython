import sys
import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pytest

# Add backend directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Remove any mocked model_bayesian from sys.modules to ensure real module is imported
# This is needed because test_main.py mocks app.model_bayesian at module level
sys.modules.pop('app.model_bayesian', None)
sys.modules.pop('app.model_functions', None)

# Now import the real module
from app.model_bayesian import BayesianCategoricalDosageClassifier


class TestBayesianCategoricalDosageClassifierInitialization:
	"""Test BayesianCategoricalDosageClassifier initialization"""

	def test_init_default_parameters(self):
		"""Test initialization with default parameters"""
		model = BayesianCategoricalDosageClassifier()
		assert model.draws == 1000
		assert model.tune == 1000
		assert model.chains == 4
		assert model.target_accept == 0.95
		assert isinstance(model.random_seed, int)
		assert model.cores == 8
		assert model.use_gpu is True
		assert model.gpu_strategy == 'aggressive'

	def test_init_custom_parameters(self):
		"""Test initialization with custom parameters"""
		model = BayesianCategoricalDosageClassifier(
			draws=500, tune=500, chains=2, target_accept=0.9, random_seed=42, cores=4, use_gpu=False, gpu_strategy='safe'
		)
		assert model.draws == 500
		assert model.tune == 500
		assert model.chains == 2
		assert model.target_accept == 0.9
		assert model.random_seed == 42
		assert model.cores == 4
		assert model.use_gpu is False
		assert model.gpu_strategy == 'safe'

	def test_tag_property(self):
		"""Test model tag property"""
		model = BayesianCategoricalDosageClassifier()
		assert model.tag == 'bayes_softmax3'

	def test_initial_state_none(self):
		"""Test that model state is None initially"""
		model = BayesianCategoricalDosageClassifier()
		assert model.idata is None
		assert model.feature_mean_ is None
		assert model.feature_std_ is None
		assert model._w_mean is None
		assert model._b_mean is None


class TestBayesianCategoricalDosageClassifierFit:
	"""Test model fitting functionality"""

	def test_fit_returns_self(self):
		"""Test that fit returns self for method chaining"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups = np.array([0, 0, 1], dtype=np.int32)

		result = model.fit(X, y, groups)
		assert result is model

	def test_fit_sets_idata(self):
		"""Test that fit sets idata"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups = np.array([0, 0, 1], dtype=np.int32)

		model.fit(X, y, groups)
		assert model.idata is not None
		assert isinstance(model.idata, az.InferenceData)

	def test_fit_sets_feature_scaling(self):
		"""Test that fit sets feature scaling parameters"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups = np.array([0, 0, 1], dtype=np.int32)

		model.fit(X, y, groups)
		assert model.feature_mean_ is not None
		assert model.feature_std_ is not None
		assert len(model.feature_mean_) == 2
		assert len(model.feature_std_) == 2

	def test_fit_sets_posterior_means(self):
		"""Test that fit sets posterior mean parameters"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups = np.array([0, 0, 1], dtype=np.int32)

		model.fit(X, y, groups)
		assert model._w_mean is not None
		assert model._mu_b_mean is not None
		assert model._w_mean.shape == (2, 3)  # n_features x 3 classes
		assert model._mu_b_mean.shape == (3,)


class TestBayesianCategoricalDosageClassifierPredict:
	"""Test prediction functionality"""

	def test_predict_proba_shape(self):
		"""Test predict_proba returns correct shape"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y_train = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups_train = np.array([0, 0, 1], dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
		probas = model.predict_proba(X_test)

		assert probas.shape == (2, 3)
		assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1

	def test_predict_proba_without_groups(self):
		"""Test predict_proba without providing groups"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y_train = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups_train = np.array([0, 0, 1], dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
		probas = model.predict_proba(X_test, groups=None)

		assert probas.shape == (2, 3)

	def test_predict_proba_with_valid_groups(self):
		"""Test predict_proba with valid group indices uses group-specific intercepts"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y_train = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups_train = np.array([0, 0, 1], dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
		groups_test = np.array([0, 1], dtype=np.int32)
		probas = model.predict_proba(X_test, groups=groups_test)

		assert probas.shape == (2, 3)
		assert np.allclose(probas.sum(axis=1), 1.0)
		assert np.all(probas >= 0.0) and np.all(probas <= 1.0)

	def test_predict_proba_values_in_range(self):
		"""All individual probability values must be in [0, 1]"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y_train = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups_train = np.array([0, 0, 1], dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
		probas = model.predict_proba(X_test)

		assert np.all(probas >= 0.0), 'All probabilities must be >= 0'
		assert np.all(probas <= 1.0), 'All probabilities must be <= 1'

	def test_predict_proba_unfitted_raises(self):
		"""Test predict_proba raises error when model not fitted"""
		model = BayesianCategoricalDosageClassifier()
		X_test = np.array([[1.0, 2.0]], dtype=np.float32)

		with pytest.raises(RuntimeError, match='Model must be fitted'):
			model.predict_proba(X_test)

	def test_predict_class_unfitted_raises(self):
		"""predict_class should raise RuntimeError when model not fitted"""
		model = BayesianCategoricalDosageClassifier()
		X_test = np.array([[1.0, 2.0]], dtype=np.float32)

		with pytest.raises(RuntimeError, match='Model must be fitted'):
			model.predict_class(X_test)

	def test_predict_class_shape(self):
		"""Test predict_class returns correct shape"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y_train = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups_train = np.array([0, 0, 1], dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
		classes = model.predict_class(X_test)

		assert classes.shape == (2,)
		assert all(c in [0, 1, 2] for c in classes)

	def test_predict_shape(self):
		"""Test predict returns expected dosage values"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y_train = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups_train = np.array([0, 0, 1], dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
		predictions = model.predict(X_test)

		assert predictions.shape == (2,)
		assert all(0 <= p <= 2 for p in predictions)  # Expected values in [0, 2]

	def test_predict_expected_dosage_formula(self):
		"""predict() must equal E[y] = sum(c * p(c)) for c in {0, 1, 2}"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y_train = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups_train = np.array([0, 0, 1], dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		X_test = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
		probas = model.predict_proba(X_test)
		expected_dosage = (probas * np.array([0.0, 1.0, 2.0], dtype=np.float32)).sum(axis=1)
		predictions = model.predict(X_test)

		np.testing.assert_array_almost_equal(predictions, expected_dosage)


class TestBayesianCategoricalDosageClassifierPersistence:
	"""Test model saving and loading"""

	def test_save_creates_files(self):
		"""Test that save creates necessary files"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups = np.array([0, 0, 1], dtype=np.int32)
		model.fit(X, y, groups)

		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'idata': tmpdir_path / 'idata.nc', 'meta': tmpdir_path / 'meta.json'}
			extra_meta = {'dataset': 'test', 'split': 'training'}

			model.save(paths, extra_meta)

			assert paths['idata'].exists()
			assert paths['meta'].exists()

	def test_save_unfitted_raises(self):
		"""Test that save raises error when model not fitted"""
		model = BayesianCategoricalDosageClassifier()

		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'idata': tmpdir_path / 'idata.nc', 'meta': tmpdir_path / 'meta.json'}
			extra_meta = {}

			with pytest.raises(RuntimeError, match='No idata to save'):
				model.save(paths, extra_meta)

	def test_load_unfitted_raises(self):
		"""Test that load raises error when paths invalid"""
		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'idata': tmpdir_path / 'idata.nc', 'meta': tmpdir_path / 'meta.json'}

			with pytest.raises(Exception):  # FileNotFoundError or similar
				BayesianCategoricalDosageClassifier.load(paths)

	def test_save_load_roundtrip(self):
		"""Save then load restores parameters and produces identical predictions"""
		model = BayesianCategoricalDosageClassifier(draws=10, tune=10, chains=1)
		X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
		y_train = np.array([0.0, 1.0, 2.0], dtype=np.float32)
		groups_train = np.array([0, 0, 1], dtype=np.int32)
		model.fit(X_train, y_train, groups_train)

		with tempfile.TemporaryDirectory() as tmpdir:
			tmpdir_path = Path(tmpdir)
			paths = {'dir': tmpdir_path, 'idata': tmpdir_path / 'idata.nc', 'meta': tmpdir_path / 'meta.json'}
			model.save(paths, {'dataset': 'roundtrip_test'})

			loaded = BayesianCategoricalDosageClassifier.load(paths)

			# Parameters must be restored within float32 precision
			# (values round-trip through JSON as Python floats, causing sub-epsilon differences)
			np.testing.assert_allclose(loaded.feature_mean_, model.feature_mean_, rtol=1e-5)
			np.testing.assert_allclose(loaded.feature_std_, model.feature_std_, rtol=1e-5)
			np.testing.assert_allclose(loaded._w_mean, model._w_mean, rtol=1e-5)
			np.testing.assert_allclose(loaded._mu_b_mean, model._mu_b_mean, rtol=1e-5)

			# Predictions must match within float32 precision
			X_test = np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
			np.testing.assert_allclose(loaded.predict(X_test), model.predict(X_test), rtol=1e-5)
			np.testing.assert_array_equal(loaded.predict_class(X_test), model.predict_class(X_test))
